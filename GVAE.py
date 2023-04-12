import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import scanpy as sc
import squidpy as sq
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn.models.autoencoder import ARGVA, ARGA
from torch_geometric.nn.sequential import Sequential
from scipy.sparse.csgraph import laplacian
import scipy.sparse as sp
from sklearn.metrics import r2_score
import sklearn.manifold as manifold
import umap.umap_ as umap

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

#Define device based on cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Found device: {device}")
#Set training mode to true
TRAINING = True

#Make sure the plot layout works correctly
plt.rcParams.update({'figure.autolayout':True, 'savefig.bbox':'tight'})

class SAGEEncoder(nn.Module):
    """GraphSAGE-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: int
            Neural network input layer size
        hidden_1: int
            Size of the first hidden layer in the network
        hidden_2: int
            Size of the second hidden layer in the network
        latent_size: int
            Size of the latent space in the network
        aggregation_method: str
            Neighborhood aggregation method to use in the
            GraphSAGE convolutions (e.g. mean, max, lstm).

    Methods:
        forward(x, edge_index):
            Feeds input x through the encoder layers.
    """

    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """Initialization function for GraphSAGE-based encoder, constructs 2 GraphSAGE
           convolutional layers, based on the specified layer sizes and aggregation
           method.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network
            aggregation_method: str
                Neighborhood aggregation method to use in the
                GraphSAGE convolutions (e.g. mean, max, lstm).

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
        return self.hlayers(x, edge_index)

class VSAGEEncoder(nn.Module):
    """GraphSAGE-based variational encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network
        aggregation_method: Neighborhood aggregation method to use in the
                            GraphSAGE convolutions (e.g. mean, max, lstm).

    Methods:
        forward: Feeds input x through the variational encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size, aggregation_method):
        """Initialization function for GraphSAGE-based variational encoder, constructs 1 GraphSAGE
           convolutional layers, a mu and a log-std GraphSAGE convolutional layer,
           based on the specified layer sizes and aggregation method. Additionally,
           a normal distribution is intialized with mean=0, std=1.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network
            aggregation_method: str
                Neighborhood aggregation method to use in the
                GraphSAGE convolutions (e.g. mean, max, lstm).

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
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x, edge_index):
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index)
        mu = self.conv_mu(x, edge_index)
        sigma = torch.exp(self.conv_logstd(x, edge_index))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl

class GATEncoder(nn.Module):
    """Graph Attention Network-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for GAT-based encoder, constructs 2 GAT
           convolutional layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

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
        return self.hlayers(x, edge_index, weight)


class VGATEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
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
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index, weight)
        mu = self.conv_mu(x, edge_index, weight)
        sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class GCNEncoder(nn.Module):
    """Graph Convolutional Network-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for GCN-based encoder, constructs 2 GCN
           convolutional layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

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
        x = self.hlayers(x, edge_index, weight)
        return x

class VGCNEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
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
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x, edge_index, weight):
        if self.num_hidden_layers > 0:
            x = self.hlayers(x, edge_index, weight)
        mu = self.conv_mu(x, edge_index, weight)
        sigma = torch.exp(self.conv_logstd(x, edge_index, weight))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class LinearEncoder(nn.Module):
    """Linear MLP-based encoder class

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        input_size: Neural network input layer size
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds input x through the encoder layers.
    """
    def __init__(self, input_size, hidden_layers, latent_size):
        """Initialization function for Linear encoder, constructs 2
           linear layers, based on the specified layer sizes.

        Parameters:
            input_size: int
                Neural network input layer size
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

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
        return self.hlayers(x)

class VLinearEncoder(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
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
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()


    def forward(self, x):
        if self.num_hidden_layers != 0:
            x = self.hlayers(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_logstd(x))
        z = mu + sigma * self.N.sample(mu.shape)
        kl  = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z, kl


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_layers, latent_size):
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
        return self.hlayers(x)


class Decoder(nn.Module):
    """Linear decoder classself.linear2(x_hat).relu()
        if TRAINING:
            x_hat = F.dropout(x_hat, p=0.2)

    Inherits:
        nn.Module: Pytorch base class for neural networks

    Attributes:
        output_size: Output layer size of the decoder, equal to the input size of the encoder
        hidden_1: Size of the first hidden layer in the network
        hidden_2: Size of the second hidden layer in the network
        latent_size: Size of the latent space in the network

    Methods:
        forward: Feeds latent space vector for one cell z through the decoder layers.
    """
    def __init__(self, output_size, hidden_layers, latent_size):
        """Initialization function for the decoder, constructs 3 linear
           decoder layers, based on the specified layer sizes.

        Parameters:
            output_size: int
                Output size of the decoder, equal to the input size of the encoder
            hidden_1: int
                Size of the first hidden layer in the network
            hidden_2: int
                Size of the second hidden layer in the network
            latent_size: int
                Size of the latent space in the network

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

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def forward(self, x, edge_index=None, cell_id=None, weight=None):
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
def plot_latent(model, pyg_graph, anndata, cell_types, device, name, number_of_cells, celltype_key, args):
    TRAINING = False
    plt.figure()
    if args.variational:
        if args.type == 'GCN' or args.type == 'GAT':
            z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                              pyg_graph.weight.to(device))
        elif args.type == 'SAGE':
            z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
        else:
            z, kl = model.encoder(pyg_graph.expr.to(device))
        z = z.to('cpu').detach().numpy()

    else:
        if args.type == 'GCN'or args.type == 'GAT':
            z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                                      pyg_graph.weight.to(device))
        elif args.type == 'SAGE':
            z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
        else:
            z = model.encoder(pyg_graph.expr.to(device))
        z = z.to('cpu').detach().numpy()
    tsne = manifold.TSNE(n_components=2)
    tsne_z =tsne.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=tsne_z[:,0], y=tsne_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    fig = plot.get_figure()
    fig.savefig(f'tsne_latentspace_{name}.png', dpi=200)
    plt.close()


    mapper = umap.UMAP()
    umap_z = mapper.fit_transform(z[:number_of_cells,:])
    plot = sns.scatterplot(x=umap_z[:,0], y=umap_z[:,1], hue=list(anndata[:number_of_cells,:].obs[celltype_key]))
    plot.legend(fontsize=3)
    fig = plot.get_figure()
    fig.savefig(f'umap_latentspace_{name}.png', dpi=200)

def train_model(model, pyg_graph, x, cell_id, weight, args, discriminator=None):
    if args.adversarial:
        if args.variational:
            if args.type == 'GCN' or args.type == 'GAT':
                z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                                  pyg_graph.weight.to(device))
            elif args.type == 'SAGE':
                z, kl = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
            else:
                z, kl = model.encoder(pyg_graph.expr.to(device))
        else:
            if args.type == 'GCN'or args.type == 'GAT':
                z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device),
                                          pyg_graph.weight.to(device))
            elif args.type == 'SAGE':
                z = model.encoder(pyg_graph.expr.to(device), pyg_graph.edge_index.to(device))
            else:
                z = model.encoder(pyg_graph.expr.to(device))
        real = torch.sigmoid(discriminator(torch.randn_like(z[cell_id,:].float())))
        fake = torch.sigmoid(discriminator(z[cell_id,:].detach()))
        real_loss = -torch.log(real + 1e-15).mean()
        fake_loss = -torch.log(1 - fake + 1e-15).mean()
        discriminator_loss = real_loss + fake_loss
        x_hat = model.discriminator(z[cell_id, :])

    elif args.variational:
        x_hat, kl = model(pyg_graph.expr, pyg_graph.edge_index, cell_id, weight)
    else:
        x_hat = model(pyg_graph.expr, pyg_graph.edge_index, cell_id, weight)

    loss = (1/pyg_graph.expr.size(dim=1)) * ((x - x_hat)**2).sum()

    if args.variational:
        loss += (1 / pyg_graph.num_nodes) * kl
    if args.adversarial:
        loss += model.reg_loss(z[cell_id])

    if not args.adversarial:
        return loss
    else:
        return loss, discriminator_loss

@torch.no_grad()
def apply_on_dataset(model, dataset, name, celltype_key, args):
    dataset = construct_graph(dataset, args=args)
    G, isolates = convert_to_graph(dataset.obsp['spatial_distances'], dataset.X, dataset.obs[celltype_key], name, args=args)
    G = nx.convert_node_labels_to_integers(G)
    pyG_graph = pyg.utils.from_networkx(G)
    pyG_graph.to(device)

    true_expr = dataset.X
    pred_expr = np.zeros(shape=(dataset.X.shape[0], dataset.X.shape[1]))
    print(true_expr.shape, pred_expr.shape)

    _, _, _, _ = variance_decomposition(pred_expr, celltype_key)

    total_loss = 0
    for cell in tqdm(G.nodes()):
        batch = pyG_graph.clone()
        batch.expr[cell, :].fill_(0.0)
        assert batch.expr[cell, :].sum() == 0
        loss, x_hat = validate(model, batch, pyG_graph.expr[cell], cell, pyG_graph.weight, args=args)
        pred_expr[cell, :] = x_hat.cpu().detach().numpy()
        total_loss += loss
    r2 = r2_score(true_expr, pred_expr)
    print(f"R2 score: {r2}")

    dataset.obs['total_counts'] = np.sum(dataset.X, axis=1)
    print(dataset.obs['total_counts'])
    print(dataset.obs['total_counts'].shape)
    sc.pl.spatial(dataset, use_raw=False, spot_size=0.1, color=['total_counts'],
                  title="Spatial distribution of true expression",
                  save=f"true_expr_spatial_{name}_all_genes", size=1, show=False)
    dataset.X = pred_expr
    dataset.obs['total_pred'] = np.sum(dataset.X, axis=1)
    sc.pl.spatial(dataset, use_raw=False, spot_size=0.1, color=['total_pred'],
                  title='Spatial distribution of predicted expression',
                  save=f"predicted_expr_spatial_{name}_all_genes", size=1, show=False)

    dataset.layers['error'] = np.absolute(true_expr - pred_expr)
    dataset.obs['total_error'] = np.sum(dataset.layers['error'], axis=1)
    dataset.obs['relative_error'] = dataset.obs['total_error'] / dataset.obs['total_counts']
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of total prediction error',
                  save=f"total_error_spatial_{name}", color=['total_error'], size=1, show=False)
    sc.pl.spatial(dataset, layer='error', spot_size=0.1, title='Spatial distribution of relative prediction error',
                  save=f"relative_error_spatial_{name}", color=['relative_error'], size=1, show=False)

    i = 0
    for gene in dataset.var_names:
        sc.pl.spatial(dataset, use_raw=False, color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of predicted expression of {gene}',
                      save=f"predicted_expr_spatial_{name}_{gene}", size=1, show=False)
        sc.pl.spatial(dataset, layer='error', color=[gene], spot_size=0.1,
                      title=f'Spatial distribution of prediction error of {gene}',
                      save=f"error_spatial_{name}_{gene}", size=1, show=False)
        i += 1
        if i == 1:
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
    print(relative_error_per_gene)

    error_per_gene = {}
    for i, gene in enumerate(dataset.var_names):
        error_per_gene[gene] = [total_error_per_gene[i],
                                average_error_per_gene[i],
                                relative_error_per_gene[i]]

    with open(f"error_per_gene_{name}.pkl", 'wb') as f:
        pickle.dump(error_per_gene, f)

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
def validate(model, val_data, x, cell_id, weight, args, discriminator=None):
    model.eval()
    if args.adversarial:
        if args.variational:
            if args.type == 'GCN' or args.type == 'GAT':
                z, kl = model.encoder(val_data.expr.to(device), val_data.edge_index.to(device),
                                  val_data.weight.to(device))
            elif args.type == 'SAGE':
                z, kl = model.encoder(val_data.expr.to(device), val_data.edge_index.to(device))
            else:
                z, kl = model.encoder(val_data.expr.to(device))
        else:
            if args.type == 'GCN'or args.type == 'GAT':
                z = model.encoder(val_data.expr.to(device), val_data.edge_index.to(device),
                                          val_data.weight.to(device))
            elif args.type == 'SAGE':
                z = model.encoder(val_data.expr.to(device), val_data.edge_index.to(device))
            else:
                z = model.encoder(val_data.expr.to(device))
        real = torch.sigmoid(discriminator(torch.randn_like(z[cell_id,:])))
        fake = torch.sigmoid(discriminator(z[cell_id,:].detach()))
        real_loss = -torch.log(real + 1e-15).mean()
        fake_loss = -torch.log(1 - fake + 1e-15).mean()
        discriminator_loss = real_loss + fake_loss
        x_hat = model.discriminator(z[cell_id, :])

    elif args.variational:
        x_hat, kl = model(val_data.expr, val_data.edge_index, cell_id, weight)
    else:
        x_hat = model(val_data.expr, val_data.edge_index, cell_id, weight)

    loss = (1/val_data.expr.size(dim=1)) * ((x - x_hat)**2).sum()

    if args.variational:
        loss += (1 / val_data.num_nodes) * kl

    if args.adversarial:
        loss += model.reg_loss(z[cell_id])

    return float(loss), x_hat

def normalize_weights(G, args):
    sigma = 0.2
    for edge in G.edges():
        if args.normalization == 'Laplacian':
            G[edge[0]][edge[1]]['weight'] = abs(np.exp(-G[edge[0]][edge[1]]['weight']**2 / sigma**2))
        else:
            G[edge[0]][edge[1]]['weight'] = np.exp(-G[edge[0]][edge[1]]['weight']**2 / sigma**2)
    return G

def convert_to_graph(adj_mat, expr_mat, cell_types=None, name='graph', args=None):
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
    nx.set_node_attributes(G, {i: {"expr" : x, 'cell_type' : y} for i, x in enumerate(np.float32(expr_mat.toarray())) for i, y in enumerate(cell_types)})

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
        graph_summary(G, name)

    #Add cell type information to the networkx graph
    if args.prediction_mode != 'full':
        G = remove_node_attributes(G, 'cell_type')

    #Calculate the weights for each edge
    print("Weighting edges")

    if args.weight:
        G = normalize_weights(G, args)


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
    for node in G.nodes():
        cell_type = G.nodes[node]['cell_type']
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            if G.nodes[neighbor]['cell_type'] == cell_type:
                G.remove_edge(neighbor, node)
    return G

def remove_isolated_nodes(G):
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def remove_node_attributes(G, attr):
    for node in G.nodes():
        del G.nodes[node][attr]
    return G

def remove_similar_celltype_edges(G):
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

def variance_decomposition(expr, celltype_key):
    """
    Total variance consists of:
    mean expression over all cells line{y},
    and for each cell i with gene j the mean expression.

    For the intracell-type variance we need to calculate
    for each cell type the mean expression of gene j

    For intercell variance we need to calculate the mean expression overall
    for gene j over all cel types.
    """
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
    return total_var, intracell_var, intercell_var, gene_var






def normalize_adjacency_matrix(M):
    d = M.sum(axis=1).A.flatten() + 1e-7  # Get row sums as a dense array
    D_data = np.reciprocal(np.sqrt(d))
    D_row_indices = np.arange(M.shape[0], dtype=np.int32)
    D_col_indices = np.arange(M.shape[1], dtype=np.int32)
    D = sp.csr_matrix((D_data, (D_row_indices, D_col_indices)), shape=M.shape) # Calculate diagonal matrix
    N = D @ M @ D
    L = D - N
    return N, L

def plot_loss_curve(data, xlabel, name):
    plt.plot(list(data.keys()), list(data.values()))
    plt.xlabel(xlabel)
    plt.ylabel('MSE')
    plt.title("GNN-based model MSE")
    plt.savefig(name, dpi=300)
    plt.close()

def plot_val_curve(train_loss, val_loss, name):
    plt.plot(list(train_loss.keys()), list(train_loss.values()), label='training')
    plt.plot(list(val_loss.keys()), list(val_loss.values()), label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Validation curve')
    plt.legend()
    plt.savefig(name, dpi=300)
    plt.close()

def construct_graph(dataset, args):
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
    return dataset

def plot_degree(degree_dist, type='degree', graph_name=''):
    #Plot log-log scaled degree distribution
    plt.ylabel('Node frequency')
    plt.hist(degree_dist, bins=np.arange(degree_dist[0].min(), degree_dist[0].max()+1))
    plt.title('Distribution of node {}'.format(type))
    plt.savefig('degree_dist_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_degree_connectivity(conn_dict, graph_name=''):
    plt.ylabel('Average connectivity')
    plt.hist(conn_dict, bins=np.arange(np.array(list(conn_dict.keys())).min(), np.array(list(conn_dict.keys())).max()+1))
    plt.title('Average degree connectivity')
    plt.savefig('degree_con_'+graph_name+'.png', dpi=300)
    plt.close()

def plot_edge_weights(edge_dict, name):
    plot = sns.histplot(edge_dict)
    plot.set(xlabel='Weight (distance)', ylabel='Edge frequency')
    plt.title('Edge weight frequencies')
    plt.savefig('dist_weight_'+name+".png", dpi=300)
    plt.close()

def graph_summary(G, name):
    summary_dict = {}
    summary_dict['name'] = name
    summary_dict['params'] = dict(vars(args))
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
    #Plot the edge weight distribution
    edge_dist = {}
    for u,v,w in G.edges(data=True):
        w = int(w['weight'])
        if w not in edge_dist:
            edge_dist[w] = 0
        edge_dist[w] += 1
    summary_dict['edge_dist'] = edge_dist
    plot_edge_weights(edge_dist, name)
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
    summary_dict['degree_dist'] = degree_dist
    #Plot the degree distribution
    plot_degree(degree_dist, 'degree', name)

    with open(f'graph_summary_{name}.pkl', 'wb') as f:
        pickle.dump(summary_dict, f)


def read_dataset(name, args):
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

    elif args.dataset == 'merfish':
        dataset = sq.datasets.merfish("data/merfish")
        organism='mouse'
        name='mouse_merfish'

    elif args.dataset == 'seqfish':
        dataset = sq.datasets.seqfish("data/seqfish")
        organism='mouse'
        name='mouse_seqfish'
        celltype_key = 'celltype_mapped_refined'

    elif args.dataset == 'slideseq':
        dataset = sq.datsets.slideseqv2("data/slideseqv2")
        print(dataset)
        organism='mouse'
        name='mouse_slideseq'


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

def set_layer_sizes(pyg_graph, args):
    if ',' in args.hidden:
        lengths = [int(x) for x in args.hidden.split(',')]
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], lengths, args.latent
    elif args.hidden == '':
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], [], args.latent
    else:
        input_size, hidden_layers, latent_size = pyg_graph.expr.shape[1], [int(args.hidden)], args.latent
    return input_size, hidden_layers, latent_size


def retrieve_model(input_size, hidden_layers, latent_size, args):
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
    decoder = Decoder(input_size, hidden_layers, latent_size)
    #Build model
    if not args.adversarial:
        model = GAE(encoder.float(), decoder.float(), args)
    else:
        if args.variational:
            model = ARGVA(encoder.float(), discriminator.float(), decoder.float())
        else:
            model = ARGA(encoder.float(), discriminator.float(), decoder.float())
    if args.adversarial:
        return model, discriminator.float()
    else:
        return model, None

def get_optimizer_list(model, args, discriminator=None):
    opt_list = []
    #Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    opt_list.append(optimizer)
    if args.adversarial:
        discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        opt_list.append(discriminator_optimizer)
    return opt_list

def train(model, pyg_graph, optimizer_list, train_i, val_i, k, args, discriminator=None):
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
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        if args.adversarial:
            discriminator.train()
            discriminator_optimizer.zero_grad()
            total_disc_loss = 0
        total_loss_over_cells = 0
        cells = random.sample(train_i, k=k)
        batch = pyg_graph.clone()
        if args.prediction_mode == 'spatial':
            batch.expr.fill_(0.0)
            assert batch.expr.sum() < 0.1
        else:
            batch.expr.index_fill_(0, torch.tensor(cells).to(device), 0.0)
            assert batch.expr[cells, :].sum() < 0.1
        for cell in cells:
            if args.adversarial:
                loss, discriminator_loss = train_model(model, batch, pyg_graph.expr[cell],
                 cell, pyg_graph.weight, args=args, discriminator=discriminator)
            else:
                loss = train_model(model, batch, pyg_graph.expr[cell], cell, pyg_graph.weight, args=args, discriminator=discriminator)
            total_loss_over_cells += loss
            if args.adversarial:
                total_disc_loss += discriminator_loss
        cells_seen += len(cells)
        print(f"Cells seen: {cells_seen}, average MSE:{total_loss_over_cells/len(cells)}")

        total_loss_over_cells.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2.0,
                                          error_if_nonfinite=True)
        optimizer.step()

        if args.adversarial:
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

        loss_over_cells[cells_seen] = total_loss_over_cells.detach().cpu()/len(cells)
        total_val_loss = 0
        total_r2 = 0
        val_cells = random.sample(val_i, k=500)
        model.eval()
        val_batch = pyg_graph.clone()
        if args.prediction_mode == 'spatial':
            val_batch.expr.fill_(0)
            assert val_batch.expr.sum() < 0.1
        else:
            val_batch.expr.index_fill_(0, torch.tensor(val_cells).to(device), 0.0)
            assert val_batch.expr[val_cells, :].sum() < 0.1
        for cell in val_cells:
            val_loss, x_hat = validate(model, val_batch, pyg_graph.expr[cell], cell, pyg_graph.weight, args=args, discriminator=discriminator)
            total_r2 += r2_score(pyg_graph.expr[cell].cpu(), x_hat.cpu())
            total_val_loss += val_loss


        train_loss_over_epochs[epoch] = total_loss_over_cells.detach().cpu()/len(cells)
        val_loss_over_epochs[epoch] = total_val_loss/500
        print(f"Epoch {epoch}, average training loss:{train_loss_over_epochs[epoch]}, average validation loss:{val_loss_over_epochs[epoch]}")
        print(f"Validation R2: {total_r2/500}")
        r2_over_epochs[epoch] = total_r2/500
    #Save trained model
    torch.save(model, f"model_{args.type}.pt")

    return loss_over_cells, train_loss_over_epochs, val_loss_over_epochs, r2_over_epochs

@torch.no_grad()
def test(model, test_i, pyg_graph, args, discriminator=None):
    print("Testing the model...")
    test_dict = {}
    total_test_loss = 0
    total_r2_test = 0
    for cell in tqdm(random.sample(test_i, k=1000)):
        test_batch = pyg_graph.clone()
        if args.prediction_mode == 'spatial':
            test_batch.expr.fill_(0.0)
            assert test_batch.expr.sum() == 0
        test_batch.expr[cell, :].fill_(0.0)
        assert test_batch.expr[cell, :].sum() == 0
        test_loss, x_hat = validate(model, test_batch, pyg_graph.expr[cell], cell, pyg_graph.weight, args=args, discriminator=discriminator)
        total_r2_test += r2_score(pyg_graph.expr[cell].cpu(), x_hat.cpu())
        total_test_loss += test_loss


    print(f"Test loss: {total_test_loss/1000}, Test R2 {total_r2_test/1000}")
    test_dict['loss'], test_dict['r2'] = total_test_loss/1000, total_r2_test/1000
    return test_dict

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-v', "--variational", action='store_true', help="Whether to use a variational AE model", default=False)
    arg_parser.add_argument('-a', "--adversarial", action="store_true", help="Whether to use a adversarial AE model", default=False)
    arg_parser.add_argument('-d', "--dataset", help="Which dataset to use", required=True)
    arg_parser.add_argument('-e', "--epochs", type=int, help="How many training epochs to use", default=1)
    arg_parser.add_argument('-c', "--cells", type=int, default=-1,  help="How many cells to sample per epoch.")
    arg_parser.add_argument('-t', '--type', type=str, choices=['GCN', 'GAT', 'SAGE', 'Linear'], help="Model type to use (GCN, GAT, SAGE, Linear)", default='GCN')
    arg_parser.add_argument('-pm', "--prediction_mode", type=str, choices=['full', 'spatial', 'expression'], default='full', help="Prediction mode to use, full uses all information, spatial uses spatial information only, expression uses expression+spatial information only")
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

    dataset, organism, name, celltype_key = read_dataset(args.dataset, args=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Empty cuda memory
    torch.cuda.empty_cache()

    if not isinstance(dataset.X, np.ndarray):
        dataset.X = dataset.X.toarray()

    _, _, _, _ = variance_decomposition(dataset.X, celltype_key)

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
    pyg_graph.to(device)
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
    print("Plotting latent space...")
    #Plot the latent test set
    plot_latent(model, pyg_graph, dataset, list(dataset.obs[celltype_key].unique()),
                device, name=f'set_{name}_{type}_{subtype}', number_of_cells=1000, celltype_key=celltype_key, args=args)
    print("Applying model on entire dataset...")
    #Apply on dataset
    apply_on_dataset(model, dataset, 'GVAE_GCN_SeqFISH', celltype_key, args=args)
