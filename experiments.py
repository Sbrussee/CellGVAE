from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from GVAE import *
import argparse
import pickle
import random
import os
import itertools

#Set seed for reproducability
random.seed(42)


gpu_uuid = "GPU-5b3b48fd-407b-f51c-705c-e77fa81fe6f0"

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
arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE'], help="Model type to use (GCN, GAT, SAGE)", default='GCN')
arg_parser.add_argument('-pm', "--prediction_mode", type=str, choices=['full', 'spatial', 'expression'], default='expression', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression +spatial information only")
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
arg_parser.add_argument('-ex', '--experiments', type=list, help='Which experiments to run', default=[1,2,3,4,5,6])
arg_parser.add_argument('-f', '--filter', action='store_true', help='Whether to filter out non-LR genes', default=False)
arg_parser.add_argument('-ds', '--downsample', action='store_true', help='Whether to use PCA decompositions of input genes', default=False)
args = arg_parser.parse_args()

args.epochs = 200
args.cells = 100
args.graph_summary = False
args.weight = True
args.normalization = 'Normal'
args.remove_same_type_edges = False
args.remove_subtype_edges = False
args.prediction_mode = 'expression'
args.neighbors = 6
args.latent = 4
args.threshold = -1
args.variational = True
args.adversarial = True
experiments = args.experiments

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


for name in ['seqfish', 'slideseq']:
    if name == 'slideseq':
        args.downsample = True
    else:
        args.downsample = False
    args.dataset = name
    dataset, organism, name, celltype_key = read_dataset(name, args)

    if args.filter:
        dataset = only_retain_lr_genes(dataset)

    if '1' in experiments:
        #Experiment 1: Run per cell type
        #Train the model on all data
        if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
            print("Constructing graph...")
            dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp1")

        print("Converting graph to PyG format...")
        if args.weight:
            if args.downsample:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.obsm['X_pca'], dataset.obs[celltype_key], name+'_train', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_train', args=args)
        else:
            if args.downsample:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.obsm['X_pca'], dataset.obs[celltype_key], name+'_train', args=args)
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
        test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

        if args.variational:
            subtype = 'variational'
        else:
            subtype = 'non-variational'

        #Plot results
        print("Plotting training plots...")
        plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_exp1_{name}_{args.type}_{subtype}.png')
        plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_exp1_{name}_{args.type}_{subtype}.png')


        #Plot the latent test set
        plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                    device, name=f'{name}_exp1', number_of_cells=1000, celltype_key=celltype_key, args=args,
                    plot_celltypes=True)

        model = model.cpu()

    if '2' in experiments:
        r2_per_comb = {}
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
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp2")

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_exp2', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_exp2", args=args)

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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_per_comb["_".join(comb)] = test_dict['r2']

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
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_exp2_{name}_{args.type}_{var}_{adv}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_exp2_{name}_{args.type}_{var}_{adv}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'exp2_{name}_{type}_{var}_{adv}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            if args.adversarial:
                apply_on_dataset(model, dataset, f'GVAE_exp2_{name}_{args.type}_{var}_{adv}', celltype_key, args=args, discriminator=discriminator)
            else:
                apply_on_dataset(model, dataset, f'GVAE_exp2_{name}_{args.type}_{var}_{adv}', celltype_key, args=args)

            model = model.cpu()

        with open("exp2.pkl", 'wb') as file:
            pickle.dump(r2_per_comb, file)



    if '3' in experiments:
        r2_per_type = {}
        args.variational = True
        args.adversarial = True
        for type in ['GCN', 'GAT', 'SAGE_max', 'SAGE_mean']:
            if type == 'SAGE_max':
                args.type = 'SAGE'
                args.aggregation_method = 'max'
            elif type == 'SAGE_mean':
                args.type = 'SAGE'
                args.aggregation_method = 'mean'
            elif type == 'GAT':
                args.type = 'GAT'
            elif type == 'GCN':
                args.type = 'GCN'

            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp3")

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_exp3', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_exp3", args=args)

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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_per_type[type] = test_dict['r2']
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
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_exp3{name}_{args.type}_{var}_{adv}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_exp3{name}_{type}_{var}_{adv}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'set_{name}_{args.type}_{var}_{adv}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, dataset, f'GVAE_exp3_{name}_{args.type}_{var}_{adv}', celltype_key, args=args, discriminator=discriminator)

            model = model.cpu()
        with open("exp3.pkl", 'wb') as file:
            pickle.dump(r2_per_type, file)

    if '4' in experiments:
        r2_per_prediction_mode = {}
        args.variational = True
        args.adversarial = True
        for prediction_mode in ['full', 'spatial', 'expression', 'spatial+expression']:
            if prediction_mode == 'full':
                args.prediction_mode = 'full'
                args.type = 'GCN'
            elif prediction_mode == 'spatial':
                args.prediction_mode = 'spatial'
                args.type = 'GCN'
            elif prediction_mode == 'expression':
                args.prediction_mode = 'expression'
                args.type = 'Linear'
            elif prediction_mode == 'spatial+expression':
                args.prediction_mode = 'spatial'
                args.type = 'GCN'
            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp4")

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_exp4', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_exp4", args=args)

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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_per_prediction_mode[prediction_mode] = test_dict['r2']

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
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_exp4_{name}_{args.type}_{prediction_mode}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_exp4_{name}_{args.type}_{prediction_mode}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'exp4_{name}_{args.type}_{prediction_mode}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, dataset, f'GVAE_exp4_{name}_{args.type}_{prediction_mode}', celltype_key, args=args, discriminator=discriminator)

            model = model.cpu()
        with open('r2_prediction_mode.pkl', 'wb') as file:
            pickle.dump(r2_per_prediction_mode, file)

    if '5' in experiments:
        r2_neighbors = {}
        for neighbors in [2,4,6,8,10]:
            args.threshold = -1
            args.neighbors = neighbors
            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp5nbs")
                dataset = spatial_analysis(dataset, celltype_key, name+"_exp5_"+str(neighbors)+"nbs")
            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_exp5nbs', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_exp5nbs", args=args)

            graph_summary(G, f'exp5_nb={neighbors}_'+name)
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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_neighbors[neighbors] = test_dict['r2']
            model = model.cpu()

        r2_thresholds = {}
        for threshold in [5, 10, 25, 50]:
            args.threshold = threshold
            args.neighbors = -1
            args.threshold = -1
            args.neighbors = neighbors
            #Train the model on all data
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name+"_exp5threshold")
                dataset = spatial_analysis(dataset, celltype_key, name+"_exp5_"+str(threshold)+"threshold")

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name+'_exp5threshold', args=args)
            else:
                G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.X, dataset.obs[celltype_key], name+"_exp5threshold", args=args)
            graph_summary(G, f'exp5_th={threshold}_'+name)
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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_thresholds[threshold] = test_dict['r2']

            model = model.cpu()

        with open("r2_neighbors.pkl", 'wb') as file:
            pickle.dump(r2_neighbors, file)


        with open("r2_thresholds.pkl", 'wb') as file:
            pickle.dump(r2_thresholds, file)

    if '6' in experiments:
        r2_filter = {}
        for filter in [True, False]:
            if filter == True:
                args.filter = True
                filter_name = "LR-filtered"
            else:
                args.filter = False
                filter_name = 'Unfiltered'
            if args.filter:
                exp6_dataset = only_retain_lr_genes(dataset)
            else:
                exp6_dataset = dataset

            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(exp6_dataset, args=args, celltype_key=celltype_key, name=name+"_exp6_"+filtername)

            print("Converting graph to PyG format...")
            if args.weight:
                G, isolates = convert_to_graph(exp6_dataset.obsp['spatial_distances'], exp6_dataset.X, exp6_dataset.obs[celltype_key], name+'_exp6_'+filtername, args=args)
            else:
                G, isolates = convert_to_graph(exp6_dataset.obsp['spatial_connectivities'], exp6_dataset.X, exp6_dataset.obs[celltype_key], name+"_exp6_"+filtername, args=args)

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
            test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

            r2_filter[filtername] = test_dict['r2']

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
            plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_exp6_{name+filtername}_{args.type}.png')
            plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_exp6_{name+filtername}_{args.type}.png')

            #Plot the latent test set
            plot_latent(model, pyg_graph, exp6_dataset, list(dataset.obs[celltype_key].unique()),
                        device, name=f'exp6_{name+filtername}_{args.type}', number_of_cells=1000, celltype_key=celltype_key, args=args)
            print("Applying model on entire dataset...")
            #Apply on dataset
            apply_on_dataset(model, exp6_dataset, f'exp6_GVAE_{name+filtername}_{args.type}', celltype_key, args=args, discriminator=discriminator)

            model = model.cpu()

    with open('r2_filter.pkl', 'wb') as file:
        pickle.dump(r2_filter, file)


    if '7' in experiments:
        organism = 'human'
        full = sc.read("/srv/scratch/chananchidas/LiverData/LiverData_RawNorm.h5ad")
        #Subset nanostring data in 4 parts
        size_obs = full.X.shape[0]
        print(f'full size {size_obs}')
        #Split by tissue type
        normal, cancer = (full[full.obs['Run_Tissue_name'] == 'NormalLiver'],
                           full[full.obs['Run_Tissue_name'] == 'CancerousLiver'])
        print(normal, cancer)
        #Delete the full dataset from memory
        del full
        fovs = np.unique(normal.obs['fov'])
        tissue = str(normal.obs['Run_Tissue_name'].unique()[0])
        for i in range(0,len(fovs)):
            fov = normal[normal.obs['fov'] == i]
            print(f"Saving {tissue} fov {i} to {i+1}...")
            print(fov.shape)
            del fov.raw
            #Save this sub-dataset
            fov.write(f'data/ns_fov_{tissue}_{i}_to_{i+1}.h5ad')

        for dataset in [f for f in os.listdir("data/") if f.startswith("ns_fov_") and 'Normal' in f]:
            data = sc.read("data/"+dataset)
            tissue = str(data.obs['Run_Tissue_name'].unique()[0])
            i = np.min(data.obs['fov'])
            if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
                print("Constructing graph...")
                dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name)

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
            train_i = [node for node in G.nodes() if node not in val_i]

            print(f"Training using fov {i} to {i+1}")
            optimizer_list = get_optimizer_list(model=model, args=args, discriminator=discriminator)
            (loss_over_cells, train_loss_over_epochs,
             val_loss_over_epochs, r2_over_epochs) = train(model, pyg_graph, optimizer_list,
                                                           train_i, val_i, k=k, args=args, discriminator=discriminator)

        latent_spaces_normal = {}
        #First get latent space for all normal tissue fovs:
        for dataset in [f for f in os.listdir("data/") if f.startswith("ns_fov_") and 'Normal' in f]:
            i = np.min(dataset.obs['fov'])
            dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name)
            G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name, args=args)
            G = nx.convert_node_labels_to_integers(G)
            pyG_graph = pyg.utils.from_networkx(G)
            pyG_graph = pyG_graph.to(device)

            latent_spaces_normal[i] = get_latent_space_vectors(model, pyG_graph, dataset, device, args)


        #Now that we trained on the normal data, score all cancer fov's
        fovs = np.unique(cancer.obs['fov'])
        tissue = str(cancer.obs['Run_Tissue_name'].unique()[0])
        for i in range(0,len(fovs)):
            fov = cancer[cancer.obs['fov'] == i]
            print(f"Saving {tissue} fov {i} to {i+1}...")
            print(fov.shape)
            del fov.raw
            #Save this sub-dataset
            fov.write(f'data/ns_fov_{tissue}_{i}_to_{i+1}.h5ad')

        latent_spaces_cancer = {}
        for dataset in [f for f in os.listdir("data/") if f.startswith("ns_fov_") and 'Cancer' in f]:
            i = np.min(dataset.obs['fov'])
            dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name)
            G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name, args=args)
            G = nx.convert_node_labels_to_integers(G)
            pyG_graph = pyg.utils.from_networkx(G)
            pyG_graph = pyG_graph.to(device)

            latent_spaces_cancer[i] = get_latent_space_vectors(model, pyG_graph, dataset, device, args)

        average_latent_space_normal = np.mean(latent_spaces_normal)
        average_latent_space_cancer = np.mean(latent_spaces_cancer)


        print(latent_spaces_normal)
        print(latent_spaces_normal.shape)
        print(latent_spaces_cancer)
        print(latent_spaces_cancer.shape)



#Use at least 2 datasets
#Evaluate using R2, MAE
#Plot the error for genes, cells and the distance between the latent spaces in terms of Eucledian/Cosine/Wasserstein and KL divergence.

#Experiment 1: Run per cell type
#split per cel type
#Experiment 2: Adversarial vs Variational vs Non Variational
#Experiment 3: GCN vs SAGE vs GAT
#Experiment 4: Linear vs. Spatial vs. no celltype vs. Full + (PCA, TSNE)
#Experiment 5: Show how the neighbors/threshold influences performance


#Experiment 6: Train on normal tissue, then calculate divergence from latent space distribution in cancer tissue. This will give a similarity score for each FOVs.
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
