import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances, plot_slice
import logging
import argparse
import pickle
import random
import os
from GVAE import *

#Set seed for reproducability
random.seed(42)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False)
arg_parser.add_argument('-a', "--adversarial", action="store_true", help="Whether to use a adversarial AE model", default=False)
arg_parser.add_argument('-d', "--dataset", help="Which dataset to use", required=False)
arg_parser.add_argument('-e', "--epochs", type=int, help="How many training epochs to use", default=1)
arg_parser.add_argument('-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch.")
arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE'], help="Model type to use (GCN, GAT, SAGE)", default='GCN')
arg_parser.add_argument('-pm', "--prediction_mode", type=str, choices=['full', 'spatial', 'expression'], default='expression', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression information only")
arg_parser.add_argument('-w', '--weight', action='store_true', help="Whether to use distance-weighted edges")
arg_parser.add_argument('-n', '--normalization', choices=["Laplacian", "Normal", "None"], default="None", help="Adjanceny matrix normalization strategy (Laplacian, Normal, None)")
arg_parser.add_argument('-rm', '--remove_same_type_edges', action='store_true', help="Whether to remove edges between same cell types")
arg_parser.add_argument('-rms', '--remove_subtype_edges', action='store_true', help='Whether to remove edges between subtypes of the same cell')
arg_parser.add_argument('-aggr', '--aggregation_method', choices=['max', 'mean'], help='Which aggregation method to use for GraphSAGE')
arg_parser.add_argument('-th', '--threshold', type=float, help='Distance threshold to use when constructing graph. If neighbors is specified, threshold is ignored.', default=-1)
arg_parser.add_argument('-ng', '--neighbors', type=int, help='Number of neighbors per cell to select when constructing graph. If threshold is specified, neighbors are ignored.', default=-1)
arg_parser.add_argument('-ls', '--latent', type=int, help='Size of the latent space to use', default=4)
arg_parser.add_argument('-hid', '--hidden', type=str, help='Specify hidden layers', default='64,32')
arg_parser.add_argument('-gs', '--graph_summary', action='store_true', help='Whether to calculate a graph summary', default=True)
arg_parser.add_argument('-f', '--filter', action='store_true', help='Whether to filter out non-LR genes', default=False)
args = arg_parser.parse_args()

args.epochs = 400
args.cells = 50
args.graph_summary = False
args.weight = True
args.normalization = 'Normal'
args.remove_same_type_edges = False
args.remove_subtype_edges = False
args.prediction_mode = 'expression'
args.latent = 4
args.threshold = -1
args.neighbors = 6
args.dataset = 'slideseq'

print(f"Parameters {args}")
dataset, organism, name, celltype_key = read_dataset(args.dataset, args)


if args.filter:
    dataset = only_retain_lr_genes(dataset)

#Subsample to k=10000
#idx = random.sample(range(dataset.shape[0]), k=10000)
#dataset = dataset[idx, :]

# Set the UUID of the GPU you want to use
gpu_uuid = "GPU-d058c48b-633a-0acc-0bc0-a2a5f0457492"

# Set the environment variable to the UUID of the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Found device: {device}")
#Set training mode to true
TRAINING = True
#Empty cuda memory
torch.cuda.empty_cache()

torch.backends.cuda.max_split_size_mb = 1024

#if not isinstance(dataset.X, np.ndarray):
#    dataset.X = dataset.X.toarray()

#_, _, _, _ = variance_decomposition(dataset.X, celltype_key)

if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
    print("Constructing graph...")
    dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name)

print("Converting graph to PyG format...")
if args.weight:
    G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
else:
    G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train", args=args)

G = nx.convert_node_labels_to_integers(G)

print("Converting graph to PyTorch Geometric dataset...")
pyg_graph = pyg.utils.from_networkx(G)

if args.prediction_mode == 'full':
    encoder = OneHotEncoder(categories=set(nx.get_node_attributes(G, 'cell_type').values()))
    pyg_graph.expr = torch.cat(pyg_graph.expr.float(), encoder.fit_transform(pyg_graph.cell_type).toarray())

pyg_graph.expr = pyg_graph.expr.float()
pyg_graph.weight = pyg_graph.weight.float()

#Split dataset
val_i = random.sample(G.nodes(), k=1000)
test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

def objective(trial):
    torch.cuda.empty_cache()
    # define hyperparameters to optimize
    variational = trial.suggest_categorical('variational', [True, False])
    adversarial = trial.suggest_categorical('adversarial', [True, False])
    model_type = trial.suggest_categorical('model_type', ['GCN', 'GAT', 'SAGE'])
    #weight = trial.suggest_categorical('weight', [True, False])
    #normalization = trial.suggest_categorical('normalization', ["Laplacian", "Normal", "None"])
    #remove_same_type_edges = trial.suggest_categorical('remove_same_type_edges', [True, False])
    #remove_subtype_edges = trial.suggest_categorical('remove_subtype_edges', [True, False])
    if model_type == 'SAGE':
        aggregation_method = trial.suggest_categorical('aggregation_method', ['max', 'mean'])
    #threshold = trial.suggest_int('threshold', 5, 100)
    #neighbors = trial.suggest_int('neighbors', 2, 10)
    #latent = trial.suggest_int('latent', 2, 12)
    hidden = trial.suggest_categorical('hidden', ['', '32', '64,32', '128,64,32', '256,128,64,32'])

    # update argparse arguments with optimized hyperparameters
    args.variational = variational
    args.adversarial = adversarial
    args.type = model_type
    if model_type == 'SAGE':
        args.aggregation_method = aggregation_method
    args.hidden = hidden

    print("Constructing model...")
    input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph, args=args)
    model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, args=args)

    print("Model:")
    print(model)
    #Send model to GPU
    model = model.to(device)
    model = model.float()

    #Set number of nodes to sample per epoch
    if args.cells == -1:
        k = G.number_of_nodes()
    else:
        k = args.cells

    optimizer_list = get_optimizer_list(model=model,args=args, discriminator=discriminator)
    # train and evaluate model with updated hyperparameters
    (loss_over_cells, train_loss_over_epochs,
     val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                    train_i, val_i, k=k, args=args, discriminator=discriminator)

    test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

    #Send model back to the cpu
    model.cpu()
    # Optimize for the best r2 of the validation set
    return np.max(list(r2_over_epochs.values()))

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=100)
    print(study)
    with open("study.pkl", 'wb') as f:
        pickle.dump(study, f)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    fig = plot_param_importances(study)
    plt.savefig("param_imp.png", dpi=300)
    plt.close()
    fig = plot_optimization_history(study)
    plt.savefig("opt_hist.png", dpi=300)
    plt.close()
