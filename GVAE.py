#Import main libaries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scanpy as sc
import squidpy as sq
import pandas as pd
#Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.models.autoencoder import ARGVA, ARGA
from torch_geometric.nn.sequential import Sequential

#Helper functions
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import sklearn.manifold as manifold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
import umap.umap_ as umap

#Import native python libaries
import sys
import os
import os.path as osp
import requests
import tarfile
import argparse
import random
import pickle
from random import sample
from datetime import datetime
from tqdm import tqdm

#Set training mode to true
TRAINING = True

#Make sure the plot layout works correctly
plt.rcParams.update({'figure.autolayout':True, 'savefig.bbox':'tight'})
#Set torch backend to reduce GPU memory usage
torch.backends.cuda.max_split_size_mb = 1024
class SAGEEncoder(nn.Module):
    """
    Class for the GraphSAGE GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through SAGEEncoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """
        Function to intialize the GraphSAGE-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
            - aggregation_method: (str): node aggregation method to use in SAGE
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        hlayers = []
        if self.num_hidden_layers  == 0:
            hlayers.append((SAGEConv(input_size, latent_size, aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        elif self.num_hidden_layers == 1:
            hlayers.append((SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((SAGEConv(hidden_layers[0], latent_size, aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        else:
            hlayers.append((SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                hlayers.append((SAGEConv(hidden_layers[i], hidden_layers[i+1], aggr=aggregation_method), 'x, edge_index -> x'))
                hlayers.append((nn.ReLU(), 'x -> x'))
                hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method), 'x, edge_index -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index', hlayers)

    def forward(self, x, edge_index):
        """
        Neural network inference function for the SAGE-encoder.
        It feeds node attribute x and edge connectivity information edge_index
        through the hidden layers of the encoder.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity
        """
        return self.hlayers(x, edge_index)

class VSAGEEncoder(nn.Module):
    """
    Class for the variational GraphSAGE GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through SAGEEncoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """
        Function to intialize the variational GraphSAGE-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input. It also constructs a normal distribution
        and intializes mu and sigma.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
            - aggregation_method: (str): node aggregation method to use in SAGE
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = []
        if self.num_hidden_layers == 0:
            self.conv_mu = SAGEConv(input_size, latent_size, aggr=aggregation_method)
            self.conv_logstd = SAGEConv(input_size, latent_size, aggr=aggregation_method)
        elif self.num_hidden_layers == 1:
            self.hlayers.append((SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method), 'x, edge_index -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            self.conv_mu = SAGEConv(hidden_layers[0], latent_size, aggr=aggregation_method)
            self.conv_logstd = SAGEConv(hidden_layers[0], latent_size, aggr=aggregation_method)
        else:
            self.hlayers.append((SAGEConv(input_size, hidden_layers[0], aggr=aggregation_method), 'x, edge_index -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                self.hlayers.append((SAGEConv(hidden_layers[i], hidden_layers[i+1], aggr=aggregation_method), 'x, edge_index -> x'))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.conv_mu = SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method)
            self.conv_logstd = SAGEConv(hidden_layers[-1], latent_size, aggr=aggregation_method)

        if self.num_hidden_layers > 0:
            self.hlayers = Sequential('x, edge_index', self.hlayers)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda(device)
        self.N.scale = self.N.scale.cuda(device)

    def forward(self, x, edge_index):
        """
        Neural network inference function for the SAGE-encoder.
        It feeds node attribute x and edge connectivity information edge_index
        through the hidden layers of the encoder. Mu and sigma are sampled from
        the model's normal distribution. The KL-divergence is calculated using
        this mu and sigma.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity

        Returns:
            -z (tensor): Sampled latent space point
            -kl (tensor): KL-divergence
        """
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index)
        mu = self.conv_mu(x, edge_index)
        sigma = torch.exp(self.conv_logstd(x, edge_index))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl

class GATEncoder(nn.Module):
    """
    Class for the Graph Attention Network GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the GAT-encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the GAT-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        hlayers = []
        if self.num_hidden_layers  == 0:
            hlayers.append((GATConv(input_size, latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        elif self.num_hidden_layers == 1:
            hlayers.append((GATConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((GATConv(hidden_layers[0], latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        else:
            hlayers.append((GATConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                hlayers.append((GATConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
                hlayers.append((nn.ReLU(), 'x -> x'))
                hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((GATConv(hidden_layers[-1], latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)

    def forward(self, x, edge_index, weight):
        """
        Neural network inference function for the SAGE-encoder.
        It feeds node attribute x and edge connectivity information edge_index
        through the hidden layers of the encoder.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity
            -weight (PyG data): Weights for each edge
        """
        return self.hlayers(x, edge_index, weight)


class VGATEncoder(nn.Module):
    """
    Class for the Variational Graph Attention Network GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the GAT-encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the variational GAT-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input. It also constructs a normal distribution
        and intializes mu and sigma.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = []
        if self.num_hidden_layers == 0:
            self.conv_mu = GATConv(input_size, latent_size)
            self.conv_logstd = GATConv(input_size, latent_size)
        elif self.num_hidden_layers == 1:
            self.hlayers.append((GATConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            self.conv_mu = GATConv(hidden_layers[0], latent_size)
            self.conv_logstd = GATConv(hidden_layers[0], latent_size)
        else:
            self.hlayers.append((GATConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                self.hlayers.append((GATConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.conv_mu = GATConv(hidden_layers[-1], latent_size)
            self.conv_logstd = GATConv(hidden_layers[-1], latent_size)

        if self.num_hidden_layers > 0:
            self.hlayers = Sequential('x, edge_index, weight', self.hlayers)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda(device)
        self.N.scale = self.N.scale.cuda(device)

    def forward(self, x, edge_index, weight):
        """
        Neural network inference function for the variational GAT-encoder.
        It feeds node attribute x along with edge connectivity information edge_index
        and edge weghts through the hidden layers of the encoder. Mu and sigma are sampled from
        the model's normal distribution. The KL-divergence is calculated using
        this mu and sigma.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity
            -weight (PyG Data): Edge weights

        Returns:
            -z (tensor): Sampled latent space point
            -kl (tensor): KL-divergence
        """
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index, weight)
        mu = self.conv_mu(x, edge_index, weight)
        sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class GCNEncoder(nn.Module):
    """
    Class for the Graph Convolutional Network GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the GCN-encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the GCN-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        hlayers = []
        if self.num_hidden_layers  == 0:
            hlayers.append((GCNConv(input_size, latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        elif self.num_hidden_layers == 1:
            hlayers.append((GCNConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((GCNConv(hidden_layers[0], latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        else:
            hlayers.append((GCNConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                hlayers.append((GCNConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
                hlayers.append((nn.ReLU(), 'x -> x'))
                hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            hlayers.append((GCNConv(hidden_layers[-1], latent_size), 'x, edge_index, weight -> x'))
            hlayers.append((nn.ReLU(), 'x -> x'))
            hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
        self.hlayers = Sequential('x, edge_index, weight', hlayers)


    def forward(self, x, edge_index, weight):
        """
        Neural network inference function for the GCN-encoder.
        It feeds node attribute x and edge connectivity information edge_index
        through the hidden layers of the encoder.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity
            -weight (PyG data): Weights for each edge
        """
        x = self.hlayers(x, edge_index, weight)
        return x

class VGCNEncoder(nn.Module):
    """
    Class for the Variational Graph Convolutional Network GNN-encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the VGCN-encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the VGCN-encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input. It also constructs a normal distribution
        and intializes mu and sigma.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = []
        if self.num_hidden_layers == 0:
            self.conv_mu = GCNConv(input_size, latent_size)
            self.conv_logstd = GCNConv(input_size, latent_size)
        elif self.num_hidden_layers == 1:
            self.hlayers.append((GCNConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            self.conv_mu = GCNConv(hidden_layers[0], latent_size)
            self.conv_logstd = GCNConv(hidden_layers[0], latent_size)
        else:
            self.hlayers.append((GCNConv(input_size, hidden_layers[0]), 'x, edge_index, weight -> x'))
            self.hlayers.append((nn.ReLU(), 'x -> x'))
            self.hlayers.append((nn.Dropout(p=0.2), 'x -> x'))
            for i in range(len(hidden_layers)-1):
                self.hlayers.append((GCNConv(hidden_layers[i], hidden_layers[i+1]), 'x, edge_index, weight -> x'))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.conv_mu = GCNConv(hidden_layers[-1], latent_size)
            self.conv_logstd = GCNConv(hidden_layers[-1], latent_size)

        if self.num_hidden_layers > 0:
            self.hlayers = Sequential('x, edge_index, weight', self.hlayers)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda(device)
        self.N.scale = self.N.scale.cuda(device)

    def forward(self, x, edge_index, weight):
        """
        Neural network inference function for the variational GCN-encoder.
        It feeds node attribute x along with edge connectivity information edge_index
        and edge weghts through the hidden layers of the encoder. Mu and sigma are sampled from
        the model's normal distribution. The KL-divergence is calculated using
        this mu and sigma.

        Parameters:
            -x (PyG Data): Node attribute
            -edge_index (PyG Data): Edge connectivity
            -weight (PyG Data): Edge weights

        Returns:
            -z (tensor): Sampled latent space point
            -kl (tensor): KL-divergence
        """
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index, weight)
        mu = self.conv_mu(x, edge_index, weight)
        sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class LinearEncoder(nn.Module):
    """
    Class for the Linear (MLP) encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the Linear encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the Linear encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = nn.Sequential()
        if self.num_hidden_layers == 0:
            self.hlayers.append(nn.Linear(input_size, latent_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        elif self.num_hidden_layers == 1:
            self.hlayers.append(nn.Linear(input_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[0], latent_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        else:
            self.hlayers.append(nn.Linear(input_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            for i in range(len(hidden_layers)-1):
                self.hlayers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[-1], latent_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))

    def forward(self, x):
        """
        Neural network inference function for the Linear encoder.
        It feeds node attribute x through the hidden layers of the encoder.

        Parameters:
            -x (PyG Data): Node attribute
        """
        return self.hlayers(x)

class VLinearEncoder(nn.Module):
    """
    Class for the Variational Linear (MLP) encoder.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the Linear encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the Variational linear encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input. It also constructs a normal distribution
        and intializes mu and sigma.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = nn.Sequential()
        if self.num_hidden_layers == 0:
            self.linear_mu = nn.Linear(input_size, latent_size)
            self.linear_logstd = nn.Linear(input_size, latent_size)
        elif self.num_hidden_layers == 1:
            self.hlayers.append(nn.Linear(input_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            self.linear_mu = nn.Linear(hidden_layers[0], latent_size)
            self.linear_logstd = nn.Linear(hidden_layers[0], latent_size)
        else:
            self.hlayers.append(nn.Linear(input_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            for i in range(len(hidden_layers)-1):
                self.hlayers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.linear_mu = nn.Linear(hidden_layers[-1], latent_size)
            self.linear_logstd = nn.Linear(hidden_layers[-1], latent_size)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda(device)
        self.N.scale = self.N.scale.cuda(device)


    def forward(self, x):
        """
        Neural network inference function for the variational GCN-encoder.
        It feeds node attribute x  through the hidden layers of the encoder. Mu and sigma are sampled from
        the model's normal distribution. The KL-divergence is calculated using
        this mu and sigma.

        Parameters:
            -x (PyG Data): Node attribute

        Returns:
            -z (tensor): Sampled latent space point
            -kl (tensor): KL-divergence
        """
        if self.num_hidden_layers != 0:
            x = self.hlayers(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_logstd(x))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class Discriminator(nn.Module):
    """
    Class for the Discriminator module in the adversarial GNN architecture.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the Linear encoder in a pytorch model.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """
        Function to intialize the Discrimantor module. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input. It also constructs a normal distribution
        and intializes mu and sigma.

        Parameters:
            - input_size (int): Size of the input layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = nn.Sequential()
        if self.num_hidden_layers == 0:
            self.hlayers.append(nn.Linear(latent_size, input_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        elif self.num_hidden_layers == 1:
            self.hlayers.append(nn.Linear(latent_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[0], input_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        else:
            self.hlayers.append(nn.Linear(latent_size, hidden_layers[-1]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            for i in range(len(hidden_layers), 1, -1):
                self.hlayers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i-2]))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[0], input_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))

    def forward(self, x):
        """
        Neural network inference function for the Discriminator.
        It feeds node attribute x through the hidden layers of the encoder.

        Parameters:
            -x (PyG Data): Node attribute
        """
        return self.hlayers(x)


class Decoder(nn.Module):
    """
    Class for the Decoder in the GNN architecture.

    Parameters:
        - nn.Module: Pytorch base functionality which allows for neural network construction and inference.

    Methods:
        -forward(): called when inferencing through the Linear encoder in a pytorch model.
    """
    def __init__(self, output_size, hidden_layers, latent_size):
        """
        Function to intialize the Variational linear encoder. First inherits pytorch
        capabilities from nn.Module. Then constructs a sequence of hidden layers
        based on the hidden_layer parameter which takes the node attribute and
        edge connectivity as input.

        Parameters:
            - output_size (int): Size of the final output layer
            - hidden_layers (list): List of layer sizes of the hidden layers
            - latent_size (int): Size of the latent space layer
        """
        super().__init__()
        self.num_hidden_layers = len(hidden_layers)
        self.hlayers = nn.Sequential()
        if self.num_hidden_layers == 0:
            self.hlayers.append(nn.Linear(latent_size, output_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        elif self.num_hidden_layers == 1:
            self.hlayers.append(nn.Linear(latent_size, hidden_layers[0]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[0], output_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
        else:
            self.hlayers.append(nn.Linear(latent_size, hidden_layers[-1]))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))
            for i in range(len(hidden_layers), 1, -1):
                self.hlayers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i-2]))
                self.hlayers.append(nn.ReLU())
                self.hlayers.append(nn.Dropout(p=0.2))
            self.hlayers.append(nn.Linear(hidden_layers[0], output_size))
            self.hlayers.append(nn.ReLU())
            self.hlayers.append(nn.Dropout(p=0.2))


    def forward(self, z):
        """
        Neural network inference function for the Decoder.
        It feeds latent space point z through the decoder back to the output layer.

        Parameters:
            -z (PyG Data): Latent space point
        """
        return self.hlayers(z)


class GAE(nn.Module):
    """Graph AutoEncoder Aggregation class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        encoder: Encoder model to use
        decoder: Decoder model to use

    Methods:
        forward: Feeds input x through the encoder to retrieve latent space z and
                 feed this to the decoder to retrieve predicted expression x_hat.
    """
    def __init__(self, encoder, decoder, args):
        """Initialization function for GCN-based encoder, constructs 2 GCN
           convolutional layers, based on the specified layer sizes.

        Parameters:
            encoder: class
                Encoder model architecture to use, can be any of the encoder classes
            decoder: class
                Decoder model to use (decoder class specified above)
            args: argsparse.Namespace
                Arguments passed by the user
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x, edge_index=None, cell_id=None, weight=None):
        """
        Inference function which encodes the node attribute x, and where applicable,
        also the edge connectivity and edge weights to retrieve latent space sample z.
        After this, z is decoded to retrieve predicted expression x_hat, for a particular
        cell, as specified by cell_id. Which encoder model to use is saved in args.type.

        Parameters:
            -x (PyG Data): Node attributes
            -edge_index (PyG Data): Edge connectivity
            -cell_id (int): Index of the cell to predict the expression for
            -weight (PyG Data): Edge weights

        Returns:
            -x_hat (tensor): Predicted expression for cell with index of cell_id.
            -kl (tensor): If variational, KL-divergence score.
        """
        if self.args.variational == False:
            if self.args.type == "Linear":
                z = self.encoder(x)
            elif self.args.type == "GCN":
                z = self.encoder(x, edge_index, weight)
            elif self.args.type == 'GAT':
                z = self.encoder(x, edge_index, weight)
            elif self.args.type == 'SAGE':
                z = self.encoder(x, edge_index)
            x_hat = self.decoder(z[cell_id, :])
            return x_hat
        else:
            if self.args.type == "Linear":
                z, kl= self.encoder(x)
            elif self.args.type == "GCN":
                z, kl = self.encoder(x, edge_index, weight)
            elif self.args.type == 'GAT':
                z, kl = self.encoder(x, edge_index, weight)
            elif self.args.type == 'SAGE':
                z, kl = self.encoder(x, edge_index)
            x_hat = self.decoder(z[cell_id, :])
            return x_hat, kl


@torch.no_grad()
def plot_latent(model, pyg_graph, anndata, cell_types, device, name, number_of_cells, celltype_key, args, plot_celltypes=False):
    """
    Plots the latent space using the model given using PCA, tSNE, UMAP.
    It can optionally also plot the latent space per celtype. The mean latent
    space per celltype is also calculated and plotted.

    Parameters:
        -model (PyG model): Pytorch Geometric model to use.
        -pyg_graph (PyG data): Pytorch Geometric Dataset to use.
        -Anndata (anndata): Squidpy/Scanpy dataset to use.
        -cell_types (list): List of unique celltypes in Anndata
        -device (str): Device to use for model inference.
        -name (str): Name to use for plotting purposes
        -number_of_cells (int): Number of cells in the latent space to plot
        -celltype_key (str): Key in Anndata where the celltype labels are stored
        -args (argparse.Namespace): Arguments given by the use
        -Plot_celltypes (Boolean): Whether to plot the latent space per celltype
    """

    TRAINING = False
    pyg_graph = pyg_graph.to(device)
    plt.figure()

    #Encode pyg_graph to get latent space vectors in z.
    if args.variational:
        if args.type == 'GCN' or args.type == 'GAT':
            z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                              pyg_graph.weight)
        elif args.type == 'SAGE':
            z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
        else:
            z, kl = model.encoder(pyg_graph.expr)
        z = z.to('cpu').detach().numpy()

    else:
        if args.type == 'GCN'or args.type == 'GAT':
            z = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                                      pyg_graph.weight)
        elif args.type == 'SAGE':
            z = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
        else:
            z = model.encoder(pyg_graph.expr)
        z = z.to('cpu').detach().numpy()

    pyg_graph = pyg_graph.cpu()

    #Filter z for any nonfinite values
    z = np.where(np.isfinite(z), z, 1e-10)

    #Check if any nonfinite values are left (for debugging)
    if np.any(np.isnan(z)) or np.any(np.isinf(z)):
        print("There are nonfinite values in the array.")
    else:
        print("There are no nonfinite values in the array.")

    #Plot latent space using tSNE
    print('TSNE...')
    tsne = manifold.TSNE(n_components=2, init='random')
    tsne_z = tsne.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=tsne_z[:,0], y=tsne_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title(f"t-SNE representation of the latent space of {name}")
    fig = plot.get_figure()
    fig.savefig(f'tsne_latentspace_{name}.png', dpi=200)
    plt.close()

    #Plot latent space using UMAP
    print('UMAP..')
    mapper = umap.UMAP()
    umap_z = mapper.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=umap_z[:,0], y=umap_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    plt.xlabel("UMAP dim 1")
    plt.ylabel("UMAP dim 2")
    plt.title(f"UMAP representation of the latent space of {name}")
    fig = plot.get_figure()
    fig.savefig(f'umap_latentspace_{name}.png', dpi=200)
    plt.close()

    #Plot latent space using PCA
    print('PCA...')
    pca = PCA(n_components=2, svd_solver='full')
    transformed_data = pca.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=transformed_data[:,0], y=transformed_data[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA representation of the latent space of {name}")
    fig = plot.get_figure()
    fig.savefig(f'pca_latentspace_{name}.png', dpi=200)
    plt.close()

    #If specified, plot the latent space per celltype
    if plot_celltypes:
        #Initialize mean latent space points per celltype
        mean_pca_per_celltype = np.zeros(shape=(len(cell_types), 3))
        mean_umap_per_celltype = np.zeros(shape=(len(cell_types), 3))
        mean_tsne_per_celltype = np.zeros(shape=(len(cell_types), 3))
        #Plot per cell types
        for i, celltype in enumerate(cell_types):
            #Select all Anndata entries for the given celltype
            obs_names = anndata[anndata.obs[celltype_key] == celltype].obs_names
            idx_to_plot = anndata.obs.index.get_indexer(obs_names)
            print(idx_to_plot)

            #Remove slashes from celltype to be able to save them in correct directory.
            celltype = celltype.replace('/', '_')


            perplexity = 300
            #Make sure perplexity < n_samples
            if perplexity > idx_to_plot.size:
                perplexity = idx_to_plot.size - 1

            tsne = manifold.TSNE(n_components=2, init='random', perplexity=perplexity)
            tsne_z =tsne.fit_transform(z[idx_to_plot,:])
            plot = sns.scatterplot(x=tsne_z[:,0], y=tsne_z[:,1])
            plt.xlabel("t-SNE dim 1")
            plt.ylabel("t-SNE dim 2")
            plt.title(f"t-SNE representation of the latent space of {celltype}")
            fig = plot.get_figure()
            fig.savefig(f'tsne_latentspace_{name}_{celltype}.png', dpi=200)
            plt.close()
            mean_tsne_per_celltype[i,:2] = np.mean(tsne_z[:,:2], axis=0)
            mean_tsne_per_celltype[i, 2] = i

            mapper = umap.UMAP()
            umap_z = mapper.fit_transform(z[idx_to_plot,:])
            plot = sns.scatterplot(x=umap_z[:,0], y=umap_z[:,1])
            plt.xlabel('UMAP dim 1')
            plt.ylabel('UMAP dim 2')
            plt.title(f"UMAP representation of the latent space of {celltype}")
            fig = plot.get_figure()
            fig.savefig(f'umap_latentspace_{name}_{celltype}.png', dpi=200)
            plt.close()
            mean_umap_per_celltype[i,:2] = np.mean(umap_z[:,:2], axis=0)
            mean_umap_per_celltype[i, 2] = i

            pca = PCA(n_components=2, svd_solver='full')
            transformed_data = pca.fit_transform(z[idx_to_plot,:])
            plot = sns.scatterplot(x=transformed_data[:,0], y=transformed_data[:,1])
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"PCA decomposition of the latent space plot_latentof {celltype}")
            fig = plot.get_figure()
            fig.savefig(f'pca_latentspace_{name}_{celltype}.png', dpi=200)
            plt.close()
            mean_pca_per_celltype[i,:2] = np.mean(transformed_data[:,:2], axis=0)
            mean_pca_per_celltype[i, 2] = i


        mapping = {k : cell_types[k] for k in range(len(cell_types))}
        print(mapping)
        #Now plot the mean latent space points per celltype
        tsne_frame = pd.DataFrame(mean_tsne_per_celltype, columns=['tsne1', 'tsne2', 'celltype']).replace(mapping)
        plot = sns.scatterplot(data=tsne_frame, x='tsne1', y='tsne2', hue='celltype')
        plt.legend(prop={ "size" : 3})
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.title(f"t-SNE representation of the mean latent space per celltype")
        fig = plot.get_figure()
        fig.savefig(f'tsne_latentspace_{name}_mean_per_celltype.png', dpi=200)
        plt.close()

        umap_frame = pd.DataFrame(mean_umap_per_celltype, columns=['umap1', 'umap2', 'celltype']).replace(mapping)
        plot = sns.scatterplot(data=umap_frame, x='umap1', y='umap2', hue='celltype')
        plt.legend(prop={ "size" : 3})
        plt.xlabel('UMAP dim 1')
        plt.ylabel('UMAP dim 2')
        plt.title(f"UMAP representation of the mean latent space per celltype")
        fig = plot.get_figure()
        fig.savefig(f'umap_latentspace_{name}_mean_per_celltype.png', dpi=200)
        plt.close()

        pca_frame = pd.DataFrame(mean_pca_per_celltype, columns=['pca1', 'pca2', 'celltype']).replace(mapping)
        plot = sns.scatterplot(data=pca_frame, x='pca1', y='pca2', hue='celltype')
        plt.legend(prop={ "size" : 3})
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"PCA decomposition of the mean latent space per celltype")
        fig = plot.get_figure()
        fig.savefig(f'pca_latentspace_{name}_mean_per_celltype.png', dpi=200)
        plt.close()

def train_model(model, pyg_graph, x, cell_id, weight, args, discriminator=None):
    """
    Function which calculates the loss between of the model between its output
    and the original dataset. It can calculate both the loss for the model itself
    and for the Discriminator module, if applicable.

    model (PyG model): Pytorch geometric model to use
    pyg_graph (PyG Data): Batch to train on
    x: Ground truth expression of cell with index cell_id
    cell_id: Index of cell to predict for
    weight: Edge weights
    args: CL arguments given by user
    discriminator: discrimantor module, when using adversarial core model.

    Returns:
        -loss: Loss for model
        Optionally:
        -discriminator_loss: Loss for discrimantor module

    """
    #Encode batch to get latent space vector z.
    pyg_graph = pyg_graph.to(device)
    if args.adversarial:
        if args.variational:
            if args.type == 'GCN' or args.type == 'GAT':
                z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                                  pyg_graph.weight)
            elif args.type == 'SAGE':
                z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
            else:
                z, kl = model.encoder(pyg_graph.expr)
        else:
            if args.type == 'GCN'or args.type == 'GAT':
                z = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                                          pyg_graph.weight)
            elif args.type == 'SAGE':
                z = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
            else:
                z = model.encoder(pyg_graph.expr)
        #Calculate discriminator loss
        real = torch.sigmoid(discriminator(torch.randn_like(z[cell_id,:].float())))
        fake = torch.sigmoid(discriminator(z[cell_id,:].detach()))
        real_loss = -torch.log(real + 1e-15).mean()
        fake_loss = -torch.log(1 - fake + 1e-15).mean()
        discriminator_loss = real_loss + fake_loss
        x_hat = model.discriminator(z[cell_id, :])
        del real
        del fake
        del real_loss
        del fake_loss

    #Decode z to get predicted expression x_hat
    elif args.variational:
        x_hat, kl = model(pyg_graph.expr, pyg_graph.edge_index, cell_id, pyg_graph.weight)
    else:
        x_hat = model(pyg_graph.expr, pyg_graph.edge_index, cell_id, pyg_graph.weight)

    #Calculate MSE loss
    loss = (1/pyg_graph.expr.size(dim=1)) * ((x.to(device) - x_hat.to(device))**2).sum()

    x = x.cpu()
    x_hat = x_hat.cpu()

    #Based on specified core model, add term to loss
    if args.variational:
        #Add normalized KL loss term
        loss += (1 / pyg_graph.num_nodes) * kl
        del kl
    if args.adversarial:
        #Add regularization loss term
        loss += model.reg_loss(z[cell_id])

    pyg_graph = pyg_graph.cpu()

    del x_hat

    if not args.adversarial:
        return loss
    else:
        return loss, discriminator_loss

@torch.no_grad()
def apply_on_dataset(model, dataset, name, celltype_key, args, discriminator=None):
    """
    Function that applies the GNN-model on the entire specified dataset.

    Parameters:
        -model (PyG model): GNN-model to use
        -dataset (anndata): Squidpy/Scanpy dataset to apply the model on
        -name (str): Name for plotting and saving purposes
        -celltype_key (str): Name of key where celltype labels are stored in dataset.
        -args (argparse.Namespace): User CL arguments
        -discriminator: Discriminator module, if applicable

    """
    #Build connectivities in dataset
    dataset = construct_graph(dataset, args=args, celltype_key=celltype_key, name=name)
    #Spatial analysis of dataset
    dataset = spatial_analysis(dataset, celltype_key, name)
    #Construct networkx graph from dataset
    G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name, args=args)
    #Relabel nodes to integer indices
    G = nx.convert_node_labels_to_integers(G)
    #Construct pytorch geometric graph from the networkx graph
    pyG_graph = pyg.utils.from_networkx(G)
    pyG_graph.expr = pyG_graph.expr.float()
    pyG_graph.weight = pyG_graph.weight.float()

    #Retrieve 'true' expression values from original dataset
    true_expr = dataset.X
    if not isinstance(true_expr, np.ndarray):
        true_expr = true_expr.toarray()

    #Intialize predicted expression matrix
    pred_expr = np.zeros(shape=(dataset.X.shape[0], dataset.X.shape[1]))
    print(true_expr.shape, pred_expr.shape)

    #Apply variance decomposition
    _, _, _, _ = variance_decomposition(pred_expr, celltype_key, name)

    #Apply model to dataset
    total_loss = 0
    batch = pyG_graph.clone()
    batch.expr = batch.expr.float()
    if args.prediction_mode == 'spatial':
        batch.expr.fill_(0.0)
        assert batch.expr.sum() == 0
    batch = batch.to(device)

    total_r2_dataset = 0
    for cell in tqdm(G.nodes()):
        orig_expr = batch.expr[cell, :]
        batch.expr[cell, :].fill_(0.0)
        assert batch.expr[cell, :].sum() == 0
        loss, x_hat = validate(model, batch, pyG_graph.expr[cell].float(), cell, pyG_graph.weight.float(), args=args, discriminator=discriminator)
        pred_expr[cell, :] = x_hat.cpu().detach().numpy()
        total_loss += loss
        total_r2_dataset += r2_score(pyG_graph.expr[cell], x_hat.cpu().detach())
        batch.expr[cell, :] = orig_expr
        del loss

    batch = batch.cpu()
    del batch
    del total_loss

    #Retrieve R2 score over entire dataset
    print(f"R2 score: {total_r2_dataset/G.number_of_nodes()}")
    pyG_graph = pyG_graph.cpu()
    print(dataset.X.shape)
    print(true_expr.shape, pred_expr.shape)

    #Apply LR analysis
    ligand_receptor_analysis(dataset, pred_expr, name, celltype_key)

    #Plot the true expression spatially
    dataset.obs['total_counts'] = np.sum(true_expr, axis=1)
    print(dataset.obs['total_counts'])
    print(dataset.obs['total_counts'].shape)
    sc.pl.spatial(dataset, layer='X', spot_size=0.1, color=['total_counts'],
                  title="Spatial distribution of true expression",
                  save=f"true_expr_spatial_{name}_all_genes.png", size=1, show=False)
    plt.close()
    #Plot the predicted expression spatially
    dataset.layers['pred'] = pred_expr
    dataset.obs['total_pred'] = np.sum(dataset.layers['pred'], axis=1)
    sc.pl.spatial(dataset, layer='pred', spot_size=0.1, color=['total_pred'],
                  title='Spatial distribution of predicted expression',
                  save=f"predicted_expr_spatial_{name}_all_genes.png", size=1, show=False)
    plt.close()

    #Calculate prediction error, calculate relative prediction error
    dataset.layers['error'] = np.absolute(true_expr - pred_expr)
    dataset.obs['total_error'] = np.sum(dataset.layers['error'], axis=1)
    dataset.obs['relative_error'] = dataset.obs['total_error'] / dataset.obs['total_counts']
    #Plot the prediction error spatially
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of total prediction error',
                  save=f"total_error_spatial_{name}.png", color=['total_error'], size=1, show=False)
    plt.close()
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of relative prediction error',
                  save=f"relative_error_spatial_{name}.png", color=['relative_error'], size=1, show=False)
    plt.close()


    i = 0
    #Plot spatial predicted expression and error per gene
    for gene in dataset.var_names:
        sc.pl.spatial(dataset, use_raw=False, color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of predicted expression of {gene}',
                      save=f"predicted_expr_spatial_{name}_{gene}.png", size=1, show=False)
        plt.close()
        sc.pl.spatial(dataset, layer='error', color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of prediction error of {gene}',
                      save=f"error_spatial_{name}_{gene}.png", size=1, show=False)
        plt.close()
        i += 1
        if i == 10:
            break

    print(dataset.var_names)
    #Calculate total error for each gene
    total_error_per_gene = np.sum(dataset.layers['error'], axis=0)
    print(total_error_per_gene)
    #Get error relative to the amount of genes present
    average_error_per_gene = total_error_per_gene/dataset.shape[1]
    print(average_error_per_gene)
    #Get error relative to amount of expression for that gene over all cells
    sum_x = np.sum(dataset.X, axis=0) + 1e-9
    relative_error_per_gene = total_error_per_gene / sum_x
    relative_error_per_gene = relative_error_per_gene.flatten()
    print("Relative error per gene shape:")
    print(relative_error_per_gene.shape)

    error_per_gene = {}
    for i, gene in enumerate(dataset.var_names):
        error_per_gene[gene] = [total_error_per_gene[i],
                                average_error_per_gene[i],
                                relative_error_per_gene[i]]

    with open(f"error_per_gene_{name}.pkl", 'wb') as f:
        pickle.dump(error_per_gene, f)

    #Plot the 10 genes with highest relative error
    error_gene_df = pd.DataFrame.from_dict(error_per_gene, orient='index',
                                 columns=['total', 'average', 'relative']).sort_values(by='relative', axis=0, ascending=False)
    print(error_gene_df)
    top10 = error_gene_df.iloc[:10]
    print(top10)
    print(top10.reset_index())
    sns.barplot(data=top10.reset_index(), x='relative', y='index', label='Relative prediction error', orient='h')
    plt.xlabel('Relative prediction error')
    plt.ylabel('Gene')
    plt.legend()
    plt.savefig(f'figures/gene_error_{name}.png', dpi=300)
    plt.close()

    #Plot the error per celltype
    error_per_cell_type = {}
    for cell_type in dataset.obs[celltype_key].unique():
        total_error = np.sum(dataset[dataset.obs[celltype_key] == cell_type].obs['total_error'])
        average_error = total_error / dataset[dataset.obs[celltype_key] == cell_type].shape[0]
        error_per_cell_type[cell_type] = average_error
        print(f"{cell_type} : {average_error}")

    error_celltype_df = pd.DataFrame.from_dict(error_per_cell_type, orient='index', columns=['average_error']).sort_values(by='average_error', axis=0, ascending=False)
    sns.barplot(data=error_celltype_df.reset_index(), x='average_error', y='index',
                label='Prediction error', orient='h')
    plt.legend()
    plt.xlabel('Prediction error')
    plt.ylabel('Cell type')
    plt.savefig(f"figures/cell_type_error_{name}.png", dpi=300)
    plt.close()
    with open(f"error_per_celltype_{name}.pkl", 'wb') as f:
        pickle.dump(error_per_cell_type, f)

@torch.no_grad()
def get_latent_space_vectors(model, pyg_graph, anndata, device, args):
    """
    Function to retrieve latent space vector z using the specified model
    on the specified dataset.

    model (PyG model): Pytorch geometric model
    pyg_graph (PyG Data): Pytorch geometric dataset
    anndata (anndata): Squidpy/scanpy dataset
    device (str): Device to use for inference
    args (argparse.Namespace): User CL arguments

    Returns:
        -z: Latent space vectors for the dataset
    """
    TRAINING = False
    if args.variational:
        if args.type == 'GCN' or args.type == 'GAT':
            z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                              pyg_graph.weight)
        elif args.type == 'SAGE':
            z, kl = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
        else:
            z, kl = model.encoder(pyg_graph.expr)
        z = z.to('cpu').detach().numpy()

    else:
        if args.type == 'GCN'or args.type == 'GAT':
            z = model.encoder(pyg_graph.expr, pyg_graph.edge_index,
                                      pyg_graph.weight)
        elif args.type == 'SAGE':
            z = model.encoder(pyg_graph.expr, pyg_graph.edge_index)
        else:
            z = model.encoder(pyg_graph.expr)
        z = z.to('cpu').detach().numpy()
    return z

@torch.no_grad()
def validate(model, val_data, x, cell_id, weight, args, discriminator=None):
    """
    Function which applies the model on the validation dataset and calculates the loss.

    model (PyG model): Pytorch geometric model to use
    val_data (PyG Data): Batch to validate the data on
    x (numpy array): ground truth expression of cell with index cell_id
    weight (PyG Data): edge weights
    args (argparse.Namespace): User CL arguments
    discriminator (PyG model): Discriminator module, if applicable.

    Returns:
        -loss: Loss for the main model
        Optionally:
        -discriminator_loss: Loss for the discriminator module
    """
    model.eval()
    val_data = val_data.to(device)
    if args.adversarial:
        if args.variational:
            if args.type == 'GCN' or args.type == 'GAT':
                z, kl = model.encoder(val_data.expr, val_data.edge_index,
                                  val_data.weight)
            elif args.type == 'SAGE':
                z, kl = model.encoder(val_data.expr, val_data.edge_index)
            else:
                z, kl = model.encoder(val_data.expr)
        else:
            if args.type == 'GCN'or args.type == 'GAT':
                z = model.encoder(val_data.expr, val_data.edge_index,
                                          val_data.weight)
            elif args.type == 'SAGE':
                z = model.encoder(val_data.expr, val_data.edge_index)
            else:
                z = model.encoder(val_data.expr)
        real = torch.sigmoid(discriminator(torch.randn_like(z[cell_id,:])))
        fake = torch.sigmoid(discriminator(z[cell_id,:].detach()))
        real_loss = -torch.log(real + 1e-15).mean()
        fake_loss = -torch.log(1 - fake + 1e-15).mean()
        discriminator_loss = real_loss + fake_loss
        x_hat = model.discriminator(z[cell_id, :])

        del real
        del fake
        del real_loss
        del fake_loss
        del discriminator_loss

    elif args.variational:
        x_hat, kl = model(val_data.expr, val_data.edge_index, cell_id, val_data.weight)
    else:
        x_hat = model(val_data.expr, val_data.edge_index, cell_id, val_data.weight)

    loss = (1/val_data.expr.size(dim=1)) * ((x.to(device) - x_hat.to(device))**2).sum()

    x = x.cpu()
    x_hat = x_hat.cpu()

    if args.variational:
        loss += (1 / val_data.num_nodes) * kl

    if args.adversarial:
        loss += model.reg_loss(z[cell_id])

    del x
    val_data = val_data.cpu()
    return float(loss), x_hat

def normalize_weights(G, args):
    """
    Function which normalizes the weights based on parameter sigma.
    We calculate the weight for two nodes u and v in G as follows:

    exp(-dist(u,v)**2 / sigma**2)

    where dist(u,v) is the distance between u and v.
    When using a Laplacian-normalized adjacency matrix, we take the absolute
    weight, because otherwise negative values may occur.

    Parameters:
        -G (networkx graph): Graph based on cell proximity
        -args (argparse.Namespace): User CL arguments

    Returns:
        -G (networkx graph): Graph with normalized weights
    """
    sigma = 0.2
    for edge in G.edges():
        if args.normalization == 'Laplacian':
            G[edge[0]][edge[1]]['weight'] = abs(np.exp(-G[edge[0]][edge[1]]['weight']**2 / sigma**2))
        else:
            G[edge[0]][edge[1]]['weight'] = np.exp(-G[edge[0]][edge[1]]['weight']**2 / sigma**2)
    return G

def convert_to_graph(adj_mat, expr_mat, cell_types=None, name='graph', args=None):
    """
    Function which converts a squidpy dataset into a networkx graph, which can
    then be used as input for the GNN autoencoder.

    Parameters:
        -adj_mat (scipy sparse array): Adjacency matrix of the spatial connectivity of the dataset
        -expr_mat (np array): Expression matrix (cells X genes)
        -cell_types (list): List of unique celltypes in the dataset
        -name (str): Name for saving and plotting
        -args (argparse.Namespace): User CL arguments

    Returns:
        -G: Networkx graph based on the squidpy dataset
        -isolates: List of isolate nodes in G, if there are any present.
    """
    if args.normalization == 'Normal' or args.normalization == 'Laplacian':
        print("Normalizing adjacency matrix...")
        N, L = normalize_adjacency_matrix(adj_mat)
        if args.normalization == 'Normal':
            G = nx.from_scipy_sparse_array(N)
        else:
            G = nx.from_scipy_sparse_array(L)

    else:
        #Make graph from adjanceny matrix
        G = nx.from_scipy_sparse_array(adj_mat)

    print("Setting node attributes")
    if not isinstance(expr_mat, np.ndarray):
        # Convert the matrix to a numpy array
        expr_mat = expr_mat.toarray()

    nx.set_node_attributes(G, {i: {"expr" : x, 'cell_type' : y} for i, x in enumerate(np.float32(expr_mat)) for i, y in enumerate(cell_types)})

    #Remove edges between same-type nodes
    if args.remove_same_type_edges:
        print("Removing same cell type edges...")
        G = remove_same_cell_type_edges(G)

    #Remove edges between subtypes of the same cell type
    if args.remove_subtype_edges:
        print("Removing edges between similar cell types")
        G = remove_similar_celltype_edges(G)

    #If any isolated nodes present, remove them:
    isolates = list(nx.isolates(G))
    print(f"Removed {len(isolates)} isolate cells")
    G = remove_isolated_nodes(G)

    #Calculate a graph statistics summary
    if args.graph_summary:
        print("Calculating graph statistics...")
        graph_summary(G, name, args)

    #Add cell type information to the networkx graph
    if args.prediction_mode != 'full':
        G = remove_node_attributes(G, 'cell_type')

    #Calculate the weights for each edge
    print("Weighting edges")

    #Weight edges
    if args.weight:
        G = normalize_weights(G, args)

    #Plot the edge weight distribution
    edge_dist = {}
    for u,v,w in G.edges(data=True):
        w = int(w['weight'])
        if w not in edge_dist:
            edge_dist[w] = 0
        edge_dist[w] += 1
    plot_edge_weights(edge_dist, name)

    #Check graph
    print(G)
    for node in G.nodes:
        print(node)
        break
    for e in G.edges():
        print(G[e[0]][e[1]])
        break
    return G, isolates

def remove_same_cell_type_edges(G):
    """
    Function which removes edges between nodes of the same celltype.

    Parameters:
        -G: Networkx graph

    Returns:
        -G: Networkx graph, with removed edges
    """
    for node in G.nodes():
        cell_type = G.nodes[node]['cell_type']
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if G.nodes[neighbor]['cell_type'] == cell_type:
                G.remove_edge(neighbor, node)
    return G

def remove_isolated_nodes(G):
    """
    Function which removes nodes which are not connected via any edges.

    Parameters:
        -G: Networkx graph

    Returns:
        -G: Networkx graph, without isolate nodes.
    """
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def remove_node_attributes(G, attr):
    """
    Function which removes the specified attribute from all nodes in G.

    Parameters:
        -G: Networkx graph
        -attr (str): Attribute to remove (e.g. celltype)

    Returns:
        -G: Networkx graph, without the specified attribute.
    """
    for node in G.nodes():
        del G.nodes[node][attr]
    return G

def remove_similar_celltype_edges(G):
    """
    Function which removes edges between nodes where the cell type of these nodes
    shares more than 3 characters on the same location in the string.

    Parameters:
        -G: Networkx Graph

    Returns:
        -G: Networkx Graph, with removed edges.

    """
    for node in G.nodes():
        cell_type = G.nodes[node]['cell_type']
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            overlap_size = 0
            for i, char in enumerate(G.nodes[neighbor]['cell_type']):
                if len(cell_type) > i:
                    if cell_type[i] == char:
                        overlap_size += 1
            if overlap_size > 3:
                G.remove_edge(neighbor, node)

    return G

def variance_decomposition(expr, celltype_key, name):
    """
    Function which decomposes the variance in an expression matrix into
        -inter-celltype variance
        -intra-celltype variance
        -gene variance

    Total variance consists of:
    mean expression over all cells line{y},
    and for each cell i with gene j the mean expression.

    For the intracell-type variance we need to calculate
    for each cell type the mean expression of gene j

    For intercell variance we need to calculate the mean expression overall
    for gene j over all cel types.

    Parameters:
        -expr (scipy sparse matrix): Expression matrix (cells X genes)
        -celltype_key (str): Key for the celltype labels.
        -name (str): name for saving and plotting

    Returns:
        -total_var: Total variance in expr.
        -intracell_var: Total intracellular variance in expr.
        -intercell_var: Total intercellular variance in expr.
        -gene_var: Total gene variance in expr.
    """

    save_dict = {}

    if not isinstance(expr, np.ndarray):
        expr = expr.toarray()

    # Add small constant value to dataset.X to avoid zero values
    expr += 0.00001

    print("Decomposing variance of dataset...")
    # Compute mean expression over all cells
    y_bar = np.mean(expr)

    y_bar_cell = np.mean(expr, axis=1)
    y_bar_gene = np.mean(expr, axis=0)

    intracell_var = np.sum(np.square(expr - y_bar_cell[:, np.newaxis]), axis=None)

    y_bar_cell_broadcast = np.broadcast_to(y_bar_cell[:, np.newaxis], expr.shape)
    intercell_var = np.sum(np.square(y_bar_cell_broadcast - y_bar_gene), axis=None)
    gene_var = np.sum(np.square(y_bar_gene - y_bar), axis=None)

    total_var = intracell_var + intercell_var + gene_var
    # Compute known total variance
    known_total_variance = np.sum((expr - y_bar)**2)

    print(f"Predicted {total_var} versus known {known_total_variance}")
    print(f"Intracell variance: {intracell_var}, fraction={intracell_var/total_var}")
    print(f"Intercell variance: {intercell_var}, fraction={intercell_var/total_var}")
    print(f"Gene variance: {gene_var}, fraction={gene_var/total_var}")

    save_dict['intracell variance'] = [intracell_var, intracell_var/total_var]
    save_dict['intercell variance'] = [intercell_var, intercell_var/total_var]
    save_dict['gene variance'] = [gene_var, gene_var/total_var]
    save_dict['total variance'] = [total_var, total_var]

    df = pd.DataFrame.from_dict(save_dict, orient='index', columns=['Variance', 'Fraction'])
    df.to_csv(f'variance_decomposition_{name}.csv')
    return total_var, intracell_var, intercell_var, gene_var


def normalize_adjacency_matrix(M):
    """
    Function which normalizes a matrix M. It outputs the normalized and
    LaPlacian normalized matrix of matrix M.

    Parameters:
        -M (scipy sparse matrix): Matrix to be normalized.

    Returns:
        -N (scipy sparse matrix): Normalized matrix
        -L (scipy sparse matrix): Laplacian matrix

    """
    d = M.sum(axis=1).A.flatten() + 1e-7  # Get row sums as a dense array
    D_data = np.reciprocal(np.sqrt(d))
    D_row_indices = np.arange(M.shape[0], dtype=np.int32)
    D_col_indices = np.arange(M.shape[1], dtype=np.int32)
    D = sp.csr_matrix((D_data, (D_row_indices, D_col_indices)), shape=M.shape) # Calculate diagonal matrix
    #Calculate normalized matrix
    N = D @ M @ D
    #Calculate laplacian matrix
    L = D - N
    return N, L

def plot_loss_curve(data, xlabel, name):
    """
    Function which plots the MSE loss over epochs or cells.

    Parameters:
        -data (dict): Data to plot, where the keys should be x-axis points (epochs, or cells)
                      and the values should be y-axis points (MSE values).
        -xlabel (str): Label to use for the x-axis.
        -name (str): Name for the file
    """
    plt.plot(list(data.keys()), list(data.values()))
    plt.xlabel(xlabel)
    plt.ylabel('MSE')
    plt.title("GNN-based model MSE")
    plt.savefig(name, dpi=300)
    plt.close()

def plot_val_curve(train_loss, val_loss, name):
    """
    Function which plots the MSE loss for both the training- and validation set
     over epochs or cells.

    Parameters:
        -train_loss (dict): Training loss data to plot, where the keys should be x-axis points (epochs, or cells)
                      and the values should be y-axis points (MSE values).
        -val_loss (dict): Validation loss data to plot, where the keys should be x-axis points (epochs, or cells)
                      and the values should be y-axis points (MSE values).
        -name (str): Name for the file
    """
    plt.plot(list(train_loss.keys()), list(train_loss.values()), label='training')
    plt.plot(list(val_loss.keys()), list(val_loss.values()), label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Validation curve')
    plt.legend()
    plt.savefig(name, dpi=300)
    plt.close()

def plot_r2_curve(r2_dict, xlabel, title, name):
    """
    Function which plots the MSE loss over epochs or cells.

    Parameters:
        -r2_dict (dict): R2-data to plot, where the keys should be x-axis points (epochs, or cells)
                      and the values should be y-axis points (R2 values).
        -xlabel (str): Label to use for the x-axis.
        -title (str): Title for the plot
        -name (str): Name for the file
    """
    plt.plot(list(r2_dict.keys()), list(r2_dict.values()))
    plt.xlabel(xlabel)
    plt.ylabel('R^2')
    plt.title(title)
    plt.savefig(name, dpi=300)
    plt.close()

def construct_graph(dataset, args, celltype_key, name=""):
    """
    Function which adds spatial connectivity information to the anndata-dataset.
    It also calculates and plots an interaction matrix.

    Parameters:
        -dataset (anndata): Squidpy/scanpy dataset.
        -args (argparse.Namespace): User CL arguments
        -celltype_key (list): List of unique celltypes in dataset.
        -name (str): Name for file naming.

    Returns:
        -dataset (anndata): Dataset with added spatial connectivity information.
    """
    if args.threshold != -1:
        threshold = args.threshold
        if args.neighbors != -1:
            sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                    radius=float(threshold), n_neighs=int(args.neighbors))
        else:
            sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                    radius=float(threshold), n_neighs=6)
    else:
        if args.neighbors != -1:
            sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                    n_neighs=int(args.neighbors))
        else:
            sq.gr.spatial_neighbors(dataset, coord_type='generic', spatial_key='spatial',
                                    radius=20, n_neighs=6)
    sq.gr.interaction_matrix(dataset, cluster_key=celltype_key)
    sq.pl.interaction_matrix(dataset, cluster_key=celltype_key, save=name+"int_matrix.png")
    plt.close()
    return dataset

def plot_degree(degree_dist, type='degree', graph_name=''):
    """
    Function which plots the degree distribution of a network.

    Parameters:
        -degree_dist (dict): Dictionary with as keys the degree and as values the node frequency.
        -type (str): String denoting the key name of degree_dist.
        -graph_name (str): Name of the graph, for saving purposes.
    """
    plt.ylabel('Node frequency')
    sns.histplot(degree_dist)
    plt.title('Distribution of node {}'.format(type))
    plt.savefig('degree_dist_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_degree_connectivity(conn_dict, graph_name=''):
    """
    Function which plots the node connectivity.

    Parameters:
        -conn_dict (dict): Dictionary with connectivity as keys and the node frequency as values.
        -graph_name (str): Name of the graph, for saving purposes.

    """
    plt.ylabel('Average connectivity')
    sns.histplot(conn_dict)
    plt.title('Average degree connectivity')
    plt.savefig('degree_con_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_edge_weights(edge_dict, name):
    """
    Function which plots the edge weight distribution in the network.

    Parameters:
        -edge_dict (dict): Dictionary with as keys the edge weights and as values the node frequency.
        -name (str): Name of the graph, for saving purposes.

    """
    plot = sns.histplot(edge_dict)
    plot.set(xlabel='Weight (distance)', ylabel='Edge frequency')
    plt.title('Edge weight frequencies')
    plt.savefig('dist_weight_'+name+".png", dpi=300)
    plt.close()

def graph_summary(G, name, args):
    """
    Function which calculates graph statistics for given graph G.

    Parameters:
        -G (networkx Graph): Graph to calculate statistics for.
        -name (str): name for file saving
        -args (argparse.Namespace): User CL arguments
    """
    summary_dict = {}
    edges = G.number_of_edges()
    summary_dict['edges'] = edges
    #get number of nodes
    nodes = G.number_of_nodes()
    summary_dict['nodes'] = nodes
    #Get density
    density = nx.density(G)
    summary_dict['density'] = density
    #Get average clustering coefficient
    clust_cf = nx.average_clustering(G)
    summary_dict['clustcf'] = clust_cf
    #Compute the average degree connectivity
    average_degree_connectivity = nx.average_degree_connectivity(G)
    summary_dict['average_degree_connectivity'] = average_degree_connectivity
    plot_degree_connectivity(average_degree_connectivity, name)
    #Compute the degree assortativity and cell type assortativity
    degree_assortativity = nx.degree_assortativity_coefficient(G)
    celltype_assortativity = nx.attribute_assortativity_coefficient(G, 'cell_type')
    summary_dict['degree_assortativity'] = degree_assortativity
    summary_dict['celltype_assortativity'] = celltype_assortativity
    #Get indegree/outdegree counts
    degrees = sorted((d for n,d in G.degree()), reverse=True)
    #Make distribution in form degree:count
    degree_dist = np.unique(degrees, return_counts=True)
    #summary_dict['degree_dist'] = degree_dist
    #Plot the degree distribution
    plot_degree(degree_dist, 'degree', name)

    df = pd.DataFrame.from_dict(summary_dict, orient='index').transpose()
    print(df.to_latex(index=False))

    with open(f'graph_summary_{name}.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)


def read_dataset(name, args):
    """
    Function which reads a dataset based on the name parameter and returns the
    anndata object of the dataset along with other important metadata of the
    dataset.

    Parameters:
        -name (str): name of the dataset to load in.
        -args (argparse.Namespace): User CL arguments

    Returns:
        -dataset (anndata): Anndata object of the dataset specified in name
        -organism (str): Organism the dataset originates from
        -name (str): Name of the dataset
        -celltype_key (str): Key for the celltype labels in the dataset.
    """
    #Get current directory, make sure output directory exists
    dirpath = os.getcwd()
    outpath = dirpath + "/output"
    if not os.path.exists(outpath):
        os.mkdir("output")

    if not os.path.exists(dirpath+"/data"):
        os.mkdir("data")

    if args.dataset == 'resolve':
        if not os.path.exists(dirpath+"/data/resolve.h5ad"):
            print("Downloading RESOLVE dataset:")
            link = requests.get("https://dl01.irc.ugent.be/spatial/adata_objects/adataA1-1.h5ad")
            with open('data/resolve.h5ad', 'wb') as f:
                f.write(link.content)
        dataset = sc.read_h5ad("data/resolve.h5ad")
        name = 'resolve'
        organism = 'mouse'
        celltype_key = 'maxScores'

    elif args.dataset == 'merfish_train':
        dataset = sq.datasets.merfish("data/merfish")
        sample = random.sample(range(dataset.n_obs), k=20000)
        dataset = dataset[sample]
        organism='mouse'
        name='mouse_merfish_train'
        celltype_key = 'Cell_class'

    elif args.dataset == 'merfish_full':
        dataset = sq.datasets.merfish("data/merfish")
        organism='mouse'
        name='mouse_merfish_full'
        celltype_key = 'Cell_class'

    elif args.dataset == 'seqfish':
        dataset = sq.datasets.seqfish("data/seqfish")
        organism='mouse'
        name='mouse_seqfish'
        celltype_key = 'celltype_mapped_refined'

    elif args.dataset == 'slideseq':
        dataset = sq.datasets.slideseqv2("data/slideseqv2")
        organism='mouse'
        name='mouse_slideseq'
        celltype_key = 'cluster'

    elif args.dataset == 'nanostring':
        dataset = sq.read.nanostring(path="data/Lung5_Rep1",
                           counts_file="Lung5_Rep1_exprMat_file.csv",
                           meta_file="Lung5_Rep1_metadata_file.csv",
                           fov_file="Lung5_Rep1_fov_positions_file.csv")
        organism = 'human'
        name= 'Lung5_Rep1'

    print("Dataset:")
    print(dataset)

    return dataset, organism, name, celltype_key

def set_layer_sizes(pyg_graph, args, panel_size):
    """
    Function which specifies the neural network layer sizes based on the user
    arguments and the panel size of the dataset.

    Parameters:
        -pyg_graph (PyG Data): Pytorch geometric dataset
        -args (argparse.Namespace): User specified arguments
        -panel_size (int): Gene panel size of the dataset in pyg_graph.

    Returns:
        -input_size (int): Size of the first input layer of the neural network
        -hidden_layers (list): List of the sizes of the hidden layers.
        -latent_size (int): Size of the latent space layer of the neural network.
        -output_size (int): Size of the output layer of the neural network.
    """
    if ',' in args.hidden:
        lengths = [int(x) for x in args.hidden.split(',')]
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], lengths, args.latent
    elif args.hidden == '':
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], [], args.latent
    else:
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], [int(args.hidden)], args.latent
    output_size = panel_size
    return input_size, hidden_layers, latent_size, output_size


def retrieve_model(input_size, hidden_layers, latent_size, output_size, args):
    """
    Function which combines neural network modules based on the specified arguments
    to build a Pytorch geometric model.

    Parameters:
        -input_size (int): Size of the input layer
        -hidden_layers (list): List of the hidden layer sizes.
        -latent_size (int): Size of the latent space layer
        -output_size (int): Size of the final output layer
        -args (argparse.Namespace): User specified arguments

    Returns:
        -model: Pytorch geometric model
        Optionally, if adversarial:
        -discriminator: Discriminator model
    """
    #Build model architecture based on given arguments
    if not args.variational and args.type == 'GCN':
        encoder = GCNEncoder(input_size, hidden_layers, latent_size)
    elif not args.variational and args.type == 'GAT':
        encoder = GATEncoder(input_size, hidden_layers, latent_size)
    elif not args.variational and args.type == 'SAGE':
        encoder = SAGEEncoder(input_size, hidden_layers, latent_size, args.aggregation_method)
    elif not args.variational and args.type == 'Linear':
        encoder = LinearEncoder(input_size, hidden_layers, latent_size)
    elif args.variational and args.type == 'GCN':
        encoder = VGCNEncoder(input_size, hidden_layers, latent_size)
    elif args.variational and args.type == 'GAT':
        encoder = VGATEncoder(input_size, hidden_layers, latent_size)
    elif args.variational and args.type == 'SAGE':
        encoder = VSAGEEncoder(input_size, hidden_layers, latent_size, args.aggregation_method)
    elif args.variational and args.type == 'Linear':
        encoder = VLinearEncoder(input_size, hidden_layers, latent_size)

    if args.adversarial:
        discriminator = Discriminator(input_size, hidden_layers, latent_size)

    #Build Decoder
    decoder = Decoder(output_size, hidden_layers, latent_size)
    #Build model
    if not args.adversarial:
        model = GAE(encoder.float(), decoder.float(), args)
    else:
        if args.variational:
            model = ARGVA(encoder.float(), discriminator.float(), decoder.float())
        else:
            model = ARGA(encoder.float(), discriminator.float(), decoder.float())
    if args.adversarial:
        return model.float(), discriminator.float()
    else:
        return model.float(), None

def get_optimizer_list(model, args, discriminator=None):
    """
    Function which retrieves a list of pytorch optimizers  based on
    the model and whether a discriminator module is present.

    Parameters:
        -model (PyG model): Pytorch geometric main model to use
        -args (argparse.Namespace): User specified arguments
        -discriminator (PyG model): Discriminator module

    Returns:
        -opt_list (list): List of optimizers for the model(s).
    """
    opt_list = []
    #Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_list.append(optimizer)
    if args.adversarial:
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        opt_list.append(discriminator_optimizer)
    return opt_list

def train(model, pyg_graph, optimizer_list, train_i, val_i, k, args, discriminator=None):
    """
    Function which contains the training loop for the pytorch geometric model.
    It also validates the model and it saves performance results.

    Parameters:
        -model (PyG model): Pytorch geometric model.
        -pyg_graph (PyG Data): Pytorch geometric dataset.
        -optimizer_list (list): Optimizer(s) to use for the model(s).
        -train_i (np array): Indices of the cells which are in the training set.
        -val_i (np array): Indices of the cells which are in the validation set.
        -k (int): Number of cells to sample per epoch.
        -args (argparse.Namespace): User specified arguments.
        Optionally:
        -discriminator (PyG model): Discriminator model to use

    Returns:
        -loss_over_cells (dict): Dictionary with cumulative number of cells trained on as keys and MSE loss as values.
        -train_loss_over_epochs (dict): Dictionary with epochs as keys and training MSE loss as values.
        -val_loss_over_epochs (dict): Dictionary with epochs as keys and validation MSE loss as values.
        -r2_over_epochs (dict): Dictionary with epochs as keys and validation R2 scores as values.

    """
    loss_over_cells = {}
    train_loss_over_epochs = {}
    val_loss_over_epochs = {}
    r2_over_epochs = {}
    cells_seen = 0

    if args.adversarial:
        optimizer, discriminator_optimizer = optimizer_list[0], optimizer_list[1]
    else:
        optimizer = optimizer_list[0]


    print("Training the model...")
    #Train for specified number of epochs
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        if args.adversarial:
            discriminator.train()
            discriminator_optimizer.zero_grad()
            total_disc_loss = 0
        total_loss_over_cells = 0
        #Sample k cells
        cells = random.sample(train_i, k=k)
        #Construct batch and change to remove expression of target cells
        batch = pyg_graph.clone()
        if args.prediction_mode == 'spatial':
            batch.expr.fill_(0.0)
            assert batch.expr.sum() < 0.1
        else:
            batch.expr.index_fill_(0, torch.tensor(cells), 0.0)
            assert batch.expr[cells, :].sum() < 0.1
        batch = batch.to(device)
        #Calculate loss for each cell
        for cell in cells:
            if args.adversarial:
                loss, discriminator_loss = train_model(model, batch, pyg_graph.expr[cell],
                 cell, pyg_graph.weight, args=args, discriminator=discriminator)
            else:
                loss = train_model(model, batch, pyg_graph.expr[cell], cell, pyg_graph.weight, args=args, discriminator=discriminator)
            total_loss_over_cells += loss
            if args.adversarial:
                total_disc_loss += discriminator_loss
        batch = batch.cpu()
        cells_seen += len(cells)
        print(f"Cells seen: {cells_seen}, average MSE:{total_loss_over_cells/len(cells)}")

        #Calculate training gradients
        total_loss_over_cells.backward()
        #Clip gradients
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0,
                                          error_if_nonfinite=False)

        #remove infinite weight parameters
        for param in model.parameters():
            if param.grad is not None:
                param.grad[~torch.isfinite(param.grad)] = 0

        #Optimize weights using gradients
        optimizer.step()

        #Learn from discriminator loss
        if args.adversarial:
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            del discriminator_loss

        loss_over_cells[cells_seen] = total_loss_over_cells.detach().cpu()/len(cells)
        total_val_loss = 0
        total_r2 = 0
        val_cells = random.sample(val_i, k=500)
        #Free up memory
        del batch
        del loss

        #Evaluate using validation dataset
        with torch.no_grad():
            model.eval()
            val_batch = pyg_graph.clone()
            val_batch = val_batch.to(device)
            if args.prediction_mode == 'spatial':
                val_batch.expr.fill_(0)
                assert val_batch.expr.sum() < 0.1
            else:
                val_batch.expr.index_fill_(0, torch.tensor(val_cells).to(device), 0.0)
                assert val_batch.expr[val_cells, :].sum() < 0.1
            for cell in val_cells:
                val_loss, x_hat = validate(model, val_batch, pyg_graph.expr[cell].to(device), cell, pyg_graph.weight.to(device), args=args, discriminator=discriminator)
                total_r2 += r2_score(pyg_graph.expr[cell], x_hat.cpu())
                total_val_loss += val_loss

            val_batch = val_batch.cpu()
            train_loss_over_epochs[epoch] = total_loss_over_cells.detach().cpu()/len(cells)
            val_loss_over_epochs[epoch] = total_val_loss/500
            print(f"Epoch {epoch}, average training loss:{train_loss_over_epochs[epoch]}, average validation loss:{val_loss_over_epochs[epoch]}")
            print(f"Validation R2: {total_r2/500}")
            r2_over_epochs[epoch] = total_r2/500

            del val_batch
            del val_loss
            del x_hat
            del total_val_loss

    model = model.cpu()
    return loss_over_cells, train_loss_over_epochs, val_loss_over_epochs, r2_over_epochs, model

@torch.no_grad()
def test(model, test_i, pyg_graph, args, discriminator=None, device=None):
    """
    Function which applies the specified autoencoder model on the test set.
    Parameters:
        -model (PyG model): Model to use.
        -test_i (np array): Array of indices for the cells in the test set.
        -args (argparse.Namespace): User specified arguments
        optionally:
        -discriminator (PyG model): Discriminator model
        -device (str): Device to use for model inference.

    Returns:
        -test_dict (dict): Dictionary with loss and r2 scores resulting from feeding the model the test set.
    """
    print("Testing the model...")
    test_dict = {}
    total_test_loss = 0
    total_r2_test = 0
    #Sample cells
    for cell in tqdm(random.sample(test_i, k=1000)):
        #Create test batch
        test_batch = pyg_graph.clone()
        if args.prediction_mode == 'spatial':
            test_batch.expr.fill_(0.0)
            assert test_batch.expr.sum() == 0
        test_batch.expr[cell, :].fill_(0.0)
        test_batch = test_batch.to(device)
        assert test_batch.expr[cell, :].sum() == 0
        test_loss, x_hat = validate(model.to(device), test_batch, pyg_graph.expr[cell].to(device), cell, pyg_graph.weight.to(device), args=args, discriminator=discriminator)
        total_r2_test += r2_score(pyg_graph.expr[cell], x_hat.cpu())
        total_test_loss += test_loss
        test_batch = test_batch.cpu()
        del test_batch
        del test_loss

    print(f"Test loss: {total_test_loss/1000}, Test R2 {total_r2_test/1000}")
    test_dict['loss'], test_dict['r2'] = total_test_loss/1000, total_r2_test/1000
    del total_test_loss
    return test_dict

def ligand_receptor_analysis(adata, pred_expr, name, cluster_key):
    """
    Function to predict ligand-receptor interactions for both the
    original expression in the dataset as well as the expression predicted by
    the autoencoder model. It uses a permutation test to identify interactions.

    Parameters:
        -adata (anndata): Squidpy/scanpy dataset to use
        -pred_expr (np array): Predicted expression matrix
        -name (str): Name for file saving
        -cluster_key (str): Key where cluster/celltype labels are saved in adata.

    """
    expr = adata.X
    #First calculate for original dataset
    res = sq.gr.ligrec(
        adata,
        n_perms=100,
        cluster_key=cluster_key,
        use_raw=False,
        transmitter_params={"categories": "ligand"},
        receiver_params={"categories": "receptor"},
        copy=True
    )
    try:
        selected = res['pvalues'][(res['pvalues'] < 0.01).any(axis=1)]
        selected['count'] =  selected.lt(0.001).sum(axis=1)
        sorted = selected.sort_values(by='count', ascending=False)
        sorted[:10].to_csv(f"lr_true_{name}.csv")
        sq.pl.ligrec(res, pvalue_threshold=0.001, remove_empty_interactions=True,
                     remove_nonsig_interactions=True, alpha=0.0001, means_range=(0.3, np.inf),
                      save=f"lr_true_{name}.png")
    except:
        print("Plotting ligrec failed")
    with open(f"ligrec_results_true_{name}.pkl", 'wb') as f:
        pickle.dump(res, f)

    #Then calculate using predicted expression
    adata.X = pred_expr
    res = sq.gr.ligrec(
        adata,
        n_perms=100,
        cluster_key=cluster_key,
        use_raw=False,
        transmitter_params={"categories": "ligand"},
        receiver_params={"categories": "receptor"},
        copy=True
    )
    try:
        selected = res['pvalues'][(res['pvalues'] < 0.01).any(axis=1)]
        selected['count'] =  selected.lt(0.001).sum(axis=1)
        sorted = selected.sort_values(by='count', ascending=False)
        sorted[:10].to_csv(f"lr_pred_{name}.csv")
        sq.pl.ligrec(res, pvalue_threshold=0.001, remove_empty_interactions=True,
                     remove_nonsig_interactions=True, alpha=0.0001, means_range=(0.3, np.inf),
                      save=f"lr_pred_{name}.png")
    except:
        print("Plotting ligrec failed")
    adata.X = expr
    with open(f"ligrec_results_pred_{name}.pkl", 'wb') as f:
        pickle.dump(res, f)

def spatial_analysis(adata, celltype_key, name):
    """
    Function which spatially analyzes a squidpy/scanpy dataset.

    Parameters:
        -adata (anndata): Dataset to analyze.
        -celltype_key (str): Key for adata where celltype labels are saved.
        -name (str): Name for file saving purposes.

    Returns:
        -adata (anndata): Dataset with additional spatial information saved.
    """
    sc.pl.spatial(adata, use_raw=False, spot_size=0.1, title=f'Spatial celltype distribution',
                  save=f"spatial_scatter_{name}.png", color=celltype_key, size=1, show=False)
    plt.close()
    sq.gr.nhood_enrichment(adata, cluster_key=celltype_key)
    sq.pl.nhood_enrichment(adata, cluster_key=celltype_key, method="ward", save=name+"ngb_enrichment.pdf")
    plt.close()
    """
    for celltype in np.unique(adata.obs[celltype_key]):
        sq.gr.co_occurrence(adata, cluster_key=celltype_key)
        sq.pl.co_occurrence(
            adata,
            cluster_key=celltype_key,
            clusters=celltype,
            figsize=(10, 5),
            save=name+celltype+".png"
        )
        plt.close()
    """

    mode = "L"
    sq.gr.ripley(adata, cluster_key=celltype_key, mode=mode, max_dist=500)
    sq.pl.ripley(adata, cluster_key=celltype_key, mode=mode, save=name+"_ripley.png")
    plt.close()
    return adata

def only_retain_lr_genes(anndata):
    """
    Function for filtering all non-ligand and non-receptor genes out of the
    anndata dataset. It uses the CellPhoneDB database for this.

    Parameters:
        -anndata (anndata): Squidpy/scanpy dataset to be filtered.

    Returns:
        -anndata_filtered: Filtered dataset with only LR genes.
    """
    # Load in the mouse_lr_pair.txt file as a pandas DataFrame
    lr_pairs = pd.read_csv('data/mouse_lr_pair.txt', sep='\t')

    # Extract the gene names from the 'ligand' and 'receptor' columns
    gene_names = [x for x in set(lr_pairs['ligand_gene_symbol']).union(set(lr_pairs['receptor_gene_symbol'])) if x in anndata.var_names]

    # Filter the Anndata dataset to only retain genes present in mouse_lr_pair.txt
    anndata_filtered = anndata[:, list(gene_names)]

    return anndata_filtered

def plot_r2_scores(r2_dict, param_name, name):
    """
    Function which plots R2 scores for specified parameter values.

    Parameters:
        -r2_dict (dict): Dictionary with parameter values as keys and r2 scores as values.
        -param_name (str): Name of the parameter values.
        -name (str): Name for save file.
    """
    x_axis, y_axis = list(r2_dict.keys()), list(r2_dict.values())
        # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the data
    ax.plot(x_axis, y_axis, marker='o')

    # Set the x-label and y-label
    ax.set_xlabel(f'{param_name}')
    ax.set_ylabel('R^2 Score')

    # Add a title to the plot
    ax.set_title(f'R^2 scores for varying {param_name}')

    plt.savefig(f"R2_{param_name}_{name}.png", dpi=300)
    plt.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Found device: {device}")
#Set training mode to true
TRAINING = True
#Empty cuda memory
torch.cuda.empty_cache()

torch.backends.cuda.max_split_size_mb = 1024

if __name__ == '__main__':

    #Set GPU identifier
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
    # Set the UUID of the GPU you want to use
    gpu_uuid = "GPU-d058c48b-633a-0acc-0bc0-a2a5f0457492"

    # Set the environment variable to the UUID of the GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device: {device}")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False)
    arg_parser.add_argument('-a', "--adversarial", action="store_true", help="Whether to use a adversarial AE model", default=False)
    arg_parser.add_argument('-d', "--dataset", help="Which dataset to use", required=True)
    arg_parser.add_argument('-e', "--epochs", type=int, help="How many training epochs to use", default=1)
    arg_parser.add_argument('-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch.")
    arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE'], help="Model type to use (GCN, GAT, SAGE)", default='GCN')
    arg_parser.add_argument('-pm', "--prediction_mode", type=str, choices=['full', 'spatial', 'expression'], default='expression', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression+spatial information only")
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

    dataset, organism, name, celltype_key = read_dataset(args.dataset, args=args)

    if args.filter:
        dataset = only_retain_lr_genes(dataset)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #Empty cuda memory
    torch.cuda.empty_cache()

    _, _, _, _ = variance_decomposition(dataset.X, celltype_key, name)

    if args.threshold != -1 or args.neighbors != -1 or args.dataset != 'resolve':
        print("Constructing graph...")
        dataset = construct_graph(dataset, args=args, celtype_key=celltype_key, name=name)

    print("Converting graph to PyG format...")
    if args.weight:
        G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.obsm['X_pca'], dataset.obs[celltype_key], name+'_train', args=args)
    else:
        G, isolates = convert_to_graph(dataset.obsp['spatial_connectivities'], dataset.obsm['X_pca'], dataset.obs[celltype_key], name+'_train', args=args)
    G = nx.convert_node_labels_to_integers(G)

    pyg_graph = pyg.utils.from_networkx(G)
    if args.prediction_mode == 'full':
        encoder = OneHotEncoder(categories=set(nx.get_node_attributes(G, 'cell_type').values()))
        pyg_graph.expr = torch.cat(pyg_graph.expr.float(), encoder.fit_transform(pyg_graph.cell_type).toarray())
    pyg_graph.expr = pyg_graph.expr.float()
    pyg_graph.weight = pyg_graph.weight.float()


    input_size, hidden_layers, latent_size, output_size = set_layer_sizes(pyg_graph, args=args, panel_size=dataset.n_vars)
    model, discriminator = retrieve_model(input_size, hidden_layers, latent_size, output_size, args=args)

    print("Model:")
    print(model)
    #Send model to GPU
    model = model.to(device)
    model.float()

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
     val_loss_over_epochs, r2_over_epochs, _) = train(model, pyg_graph, optimizer_list,
                                                   train_i, val_i, k=k, args=args, discriminator=discriminator)
    test_dict = test(model, test_i, pyg_graph, args=args, discriminator=discriminator, device=device)

    if args.variational:
        subtype = 'variational'
    else:
        subtype = 'non-variational'

    #Plot results
    print("Plotting training plots...")
    plot_loss_curve(loss_over_cells, 'cells', f'loss_curve_cells_{name}_{type}_{subtype}.png')
    plot_val_curve(train_loss_over_epochs, val_loss_over_epochs, f'val_loss_curve_epochs_{name}_{type}_{subtype}.png')
    plot_r2_curve(r2_over_epochs, 'epochs', 'R2 over training epochs', f'r2_curve_{name}')
    print("Plotting latent space...")
    #Plot the latent test set
    plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                device, name=f'set_{name}_{type}_{subtype}', number_of_cells=1000, celltype_key=celltype_key, args=args)
    print("Applying model on entire dataset...")

    if args.dataset == 'merfish_train':
        dataset = read_dataset('merfish_full')
    #Apply on dataset
    apply_on_dataset(model, dataset, 'GVAE_GCN_SeqFISH', celltype_key, args=args)

    model = model.cpu()
