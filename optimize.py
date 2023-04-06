import optuna
import argparse
import pickle
from GVAE import *

seqfish = read_dataset('seqfish')

def objective(trial):
    # define hyperparameters to optimize
    variational = trial.suggest_categorical('variational', [True, False])
    adversarial = trial.suggest_categorical('adversarial', [True, False])
    model_type = trial.suggest_categorical('model_type', ['GCN', 'GAT', 'SAGE', 'Linear'])
    weight = trial.suggest_categorical('weight', [True, False])
    normalization = trial.suggest_categorical('normalization', ["Laplacian", "Normal", "None"])
    add_cell_types = trial.suggest_categorical('add_cell_types', [True, False])
    remove_same_type_edges = trial.suggest_categorical('remove_same_type_edges', [True, False])
    remove_subtype_edges = trial.suggest_categorical('remove_subtype_edges', [True, False])
    aggregation_method = trial.suggest_categorical('aggregation_method', ['max', 'mean'])
    threshold = trial.suggest_int('threshold', 5, 100)
    neighbors = trial.suggest_int('neighbors', 2, 30)
    latent = trial.suggest_int('latent', 2, 12)
    hidden = trial.suggest_categorical('hidden', ['', '32', '64,32', '128,64,32', '256,128,64,32', '512,256,128,64,32'])

    epochs = 50
    cells = 1000
    prediction_mode = 'full'
    graph_summary = False

    # update argparse arguments with optimized hyperparameters
    variational = variational
    adversarial = adversarial
    type = model_type
    weight = weight
    normalization = normalization
    add_cell_types = add_cell_types
    remove_same_type_edges = remove_same_type_edges
    remove_subtype_edges = remove_subtype_edges
    aggregation_method = aggregation_method
    threshold = threshold
    neighbors = neighbors
    latent = latent
    hidden = hidden

    #Define device based on cuda availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Found device: {device}")
    #Set training mode to true
    TRAINING = True
    #Empty cuda memory
    torch.cuda.empty_cache()

    dataset = seqfish

    if not isinstance(dataset.X, np.ndarray):
        dataset.X = dataset.X.toarray()

    _, _, _, _ = variance_decomposition(dataset.X, celltype_key)

    print("Constructing graph...")
    dataset = construct_graph(dataset)

    print("Converting graph to PyG format...")
    if weight:
        G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train')
    else:
        G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train")

    G = nx.convert_node_labels_to_integers(G)

    pyg_graph = pyg.utils.from_networkx(G)

    pyg_graph.to(device)
    input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph)
    model = retrieve_model(input_size, hidden_layers, latent_size)

    print("Model:")
    print(model)
    #Send model to GPU
    model = model.to(device)
    pyg.transforms.ToDevice(device)

    #Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if adversarial:
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    #Set number of nodes to sample per epoch
    if cells == -1:
        k = G.number_of_nodes()
    else:
        k = cells

    #Split dataset
    val_i = random.sample(G.nodes(), k=1000)
    test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
    train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

    optimizer_list = get_optimizer_list()
    # train and evaluate model with updated hyperparameters
    (loss_over_cells, train_loss_over_epochs,
     val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list, train_i, val_i)

    # Optimize for the best r2 validation found
    return np.max(list(r2_over_epochs.values))

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(sampler=optune.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())
    study.optimze(objective, n_trials=100)
    print(study)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    with open("study.pkl", 'rb') as f:
        pickle.dump(study, f)
