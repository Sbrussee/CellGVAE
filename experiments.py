from sklearn.decomposition import PCA
from GVAE import *
import argparse
import pickle
import random
import os
import itertools

#Set seed for reproducability
random.seed(42)


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

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False)
arg_parser.add_argument('-a', "--adversarial", action="store_true", help="Whether to use a adversarial AE model", default=False)
arg_parser.add_argument('-d', "--dataset", help="Which dataset to use", required=False)
arg_parser.add_argument('-e', "--epochs", type=int, help="How many training epochs to use", default=1)
arg_parser.add_argument('-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch.")
arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE', 'Linear'], help="Model type to use (GCN, GAT, SAGE, Linear)", default='GCN')
arg_parser.add_argument('-pm', "--prediction_mode", type=str, choices=['full', 'spatial'], default='full', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression information only")
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
args = arg_parser.parse_args()

args.epochs = 200
args.cells = 100
args.graph_summary = False
args.weight = True
args.normalization = 'Normal'
args.remove_same_type_edges = True
args.remove_subtype_edges = False
args.prediction_mode = 'Full'
args.neighbors = 6
args.latent = 4
args.theshold = -1

def apply_pca(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    transformed_data = pca.transform(data)

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

def apply_tsne(data, perplexity=30, learning_rate=200, n_iter=1000):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
    transformed_data = tsne.fit_transform(data)

    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()


experiments = [1,2,3,4,5]


for name in ['seqfish', 'slideseqv2']:
    args.dataset = name
    dataset, organism, name, celltype_key = read_dataset(name, args)
    if 1 in experiments:
        #Experiment 1: Run per cell type
        #Train the model on all data
        if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
            print("Constructing graph...")
            dataset = construct_graph(dataset, args=args)

        print("Converting graph to PyG format...")
        if args.weight:
            G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
        else:
            G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train", args=args)

        G = nx.convert_node_labels_to_integers(G)

        pyg_graph = pyg.utils.from_networkx(G)
        pyg_graph.expr = pyg_graph.expr.float()
        pyg_graph.weight = pyg_graph.weight.float()
        input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph, args=args)
        model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, args=args)

        print("Model:")
        print(model)
        #Send model to GPU
        model = model.to(device)
        model.float()
        pyg.transforms.ToDevice(device)

        #Set number of nodes to sample per epoch
        if args.cells == -1:
            k = G.number_of_nodes()
        else:
            k = args.cells

        #Split dataset
        val_i = random.sample(G.nodes(), k=1000)
        test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
        train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

        optimizer_list = get_optimizer_list(model=model, args=args, discriminator=discriminator)
        (loss_over_cells, train_loss_over_epochs,
         val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                       train_i, val_i, k=k, args=args, discriminator=discriminator)
        test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator)

        if args.variational:
            subtype = 'variational'
        else:
            subtype = 'non-variational'

        #Plot results
        print("Plotting training plots...")
        plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{subtype}.png')
        plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{subtype}.png')


        #Plot the latent test set
        plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                    device, name=f'set_{name}', number_of_cells=1000, celltype_key=celltype_key, args=args)

    if 2 in experiments:
        core_models = ['adversarial', 'variational', 'normal']
        for comb in itertools.combinations(core_models, 2):
            if 'adversarial' in comb:
                args.adversarial = True
            else:
                args.adversarial = False
            if 'variational' in comb:
                args.variational = True
            else:
                args.variational = False

            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args)

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train", args=args)

            G = nx.convert_node_labels_to_integers(G)

            pyg_graph = pyg.utils.from_networkx(G)
            pyg_graph.expr = pyg_graph.expr.float()
            pyg_graph.weight = pyg_graph.weight.float()
            input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph, args=args)
            model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, args=args)

            print("Model:")
            print(model)
            #Send model to GPU
            model = model.to(device)
            model.float()
            pyg.transforms.ToDevice(device)

            #Set number of nodes to sample per epoch
            if args.cells == -1:
                k = G.number_of_nodes()
            else:
                k = args.cells

            #Split dataset
            val_i = random.sample(G.nodes(), k=1000)
            test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
            train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

            optimizer_list = get_optimizer_list(model=model, args=args, discriminator=discriminator)
            (loss_over_cells, train_loss_over_epochs,
             val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                           train_i, val_i, k=k, args=args, discriminator=discriminator)
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator)

            if args.variational:
                var = 'variational'
            else:
                var = 'non-variational'

            if args.adversarial:
                adv = 'adversarial'
            else:
                adv = 'non-adverserial'

            #Plot results
            print("Plotting training plots...")
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{var}_{adv}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{var}_{adv}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'set_{name}_{type}_{var}_{adv}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, dataset, f'GVAE_{name}_{type}_{var}_{adv}', celltype_key, args=args)

    if 3 in experiments:
        args.variational = False
        args.adversarial = False
        for type in ['GCN', 'GAT', 'SAGE_max', 'SAGE_avg']:
            if type == 'SAGE_max':
                args.type = 'SAGE'
                args.aggregation_method = 'max'
            elif type == 'SAGE_avg':
                args.type = 'SAGE'
                args.aggregation_method = 'avg'
            elif type == 'GAT':
                args.type = 'GAT'
            elif type == 'GCN':
                args.type = 'GCN'

            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args)

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train", args=args)

            G = nx.convert_node_labels_to_integers(G)

            pyg_graph = pyg.utils.from_networkx(G)
            pyg_graph.expr = pyg_graph.expr.float()
            pyg_graph.weight = pyg_graph.weight.float()
            input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph, args=args)
            model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, args=args)

            print("Model:")
            print(model)
            #Send model to GPU
            model = model.to(device)
            model.float()
            pyg.transforms.ToDevice(device)

            #Set number of nodes to sample per epoch
            if args.cells == -1:
                k = G.number_of_nodes()
            else:
                k = args.cells

            #Split dataset
            val_i = random.sample(G.nodes(), k=1000)
            test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
            train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

            optimizer_list = get_optimizer_list(model=model, args=args, discriminator=discriminator)
            (loss_over_cells, train_loss_over_epochs,
             val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                           train_i, val_i, k=k, args=args, discriminator=discriminator)
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator)

            if args.variational:
                var = 'variational'
            else:
                var = 'non-variational'

            if args.adversarial:
                adv = 'adversarial'
            else:
                adv = 'non-adverserial'

            #Plot results
            print("Plotting training plots...")
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{var}_{adv}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{var}_{adv}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'set_{name}_{type}_{var}_{adv}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, dataset, f'GVAE_{name}_{type}_{var}_{adv}', celltype_key, args=args)

    if 4 in experiments:
        args.variational = False
        args.adversarial = False
        for prediction_mode in ['full', 'spatial', 'expression', 'spatial+expression']:
            if prediction_mode == 'full':
                args.prediction_mode = 'full'
                args.type = 'GCN'
            elif prediction_mode == 'spatial':
                args.prediction_mode = 'spatial'
                args.type = 'GCN'
            elif prediction_mode == 'expression':
                args.prediction_mode = 'expression'
                args.type = 'linear'
            elif prediction_mode == 'spatial+expression':
                args.prediction_mode = 'spatial'
                args.type = 'GCN'
            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args)

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_train", args=args)

            G = nx.convert_node_labels_to_integers(G)

            pyg_graph = pyg.utils.from_networkx(G)
            pyg_graph.expr = pyg_graph.expr.float()
            pyg_graph.weight = pyg_graph.weight.float()
            input_size, hidden_layers, latent_size = set_layer_sizes(pyg_graph, args=args)
            model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, args=args)

            print("Model:")
            print(model)
            #Send model to GPU
            model = model.to(device)
            model.float()
            pyg.transforms.ToDevice(device)

            #Set number of nodes to sample per epoch
            if args.cells == -1:
                k = G.number_of_nodes()
            else:
                k = args.cells

            #Split dataset
            val_i = random.sample(G.nodes(), k=1000)
            test_i = random.sample([node for node in G.nodes() if node not in val_i], k=1000)
            train_i = [node for node in G.nodes() if node not in val_i and node not in test_i]

            optimizer_list = get_optimizer_list(model=model, args=args, discriminator=discriminator)
            (loss_over_cells, train_loss_over_epochs,
             val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                           train_i, val_i, k=k, args=args, discriminator=discriminator)
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator)

            if args.variational:
                var = 'variational'
            else:
                var = 'non-variational'

            if args.adversarial:
                adv = 'adversarial'
            else:
                adv = 'non-adverserial'

            #Plot results
            print("Plotting training plots...")
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{prediction_mode}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{prediction_mode}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'set_{name}_{type}_{prediction_mode}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, dataset, f'GVAE_{name}_{type}_{prediction_mode}', celltype_key, args=args)



#Use at least 2 datasets
#Evaluate using R2, MAE
#Plot the error for genes, cells and the distance between the latent spaces in terms of Eucledian/Cosine/Wasserstein and KL divergence.

#Experiment 1: Run per cell type
#split per cel type
#Experiment 2: Adversarial vs Variational vs Non Variational
#Experiment 3: GCN vs SAGE vs GAT
#Experiment 4: Linear vs. Spatial vs. no celltype vs. Full + (PCA, TSNE)


#Experiment 5: Train on normal tissue, then calculate divergence from latent space distribution in cancer tissue. This will give a similarity score for each FOVs.
#Feed each normal fov to the model
#calculate mean latent space point per cell type in fov
#calculate mean latent space point per fov

#Visualize the latent space after training on the fov slices

#Cluster this latent space of fov points to cluster fov into sections

#Now feed the malignant slices, again, calculate mean per cell type and per fov
#Score each fov on distance to fov clusters and global distribution overall.



#Extra 1: Difference between preprocessing steps in graph statistics
#Extra 2: Performance differences while changing graph construction parameters
#Extra 3: Difference in datasets in variance attribution
#Extra 4: Effect of latent space size between datasets
#Extra 5: Effect of removing edges
