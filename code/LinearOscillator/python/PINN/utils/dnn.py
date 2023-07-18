# Basic fully conneted neural networks that are shared across different architectures.

# Author: Hongli Zhao
# Date: 07/18/2023

##########################################################################################
# list of imports
##########################################################################################
import torch
import torch.nn as nn
import torch.autograd
import torch.optim
# set precision
torch.set_default_dtype(torch.float64)
# set random seed
torch.manual_seed(10)

from collections import OrderedDict
# other scientific libraries
import numpy as np
import scipy
import scipy.io

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# import utility functions
from .helpers import *


##########################################################################################
# Vanilla DNN
##########################################################################################
class DNN(torch.nn.Module):
    def __init__(
        self, layers, 
        activation=torch.nn.Tanh, 
        last_layer_activation=None,
        initialization=None
    ):
        """ 
            Custom initialization of neural network layers with the option 
            of changing the output layer's activation function.
        """
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = activation
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        if last_layer_activation is not None:
            layer_list.append(
            ('activation_%d' % (self.depth - 1), last_layer_activation())
        )

        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
        # custom initialization modes
        self.initialize(mode=initialization)
        
        
    def forward(self, x):
        return self.layers(x)
    
    def initialize(self, mode):
        if mode == None:
            return
        else:
            for layer in self.layers:
                if isinstance(layer, torch.nn.Linear):
                    # initialize depending on mode
                    if mode == "xavier":
                        torch.nn.init.xavier_uniform_(layer.weight)
                    elif mode == "kaiming":
                        torch.nn.init.kaiming_uniform_(layer.weight)
                    elif mode == "normal":
                        torch.nn.init.normal_(layer.weight)
                    elif mode == "uniform":
                        torch.nn.init.uniform_(layer.weight)
                    elif mode == "ones":
                        torch.nn.init.ones_(layer.weight)
                    else:
                        raise NotImplementedError()
            return



##########################################################################################
# Fourier-embedded DNN
##########################################################################################
class FourierEmbeddedDNN(torch.nn.Module):
    def __init__(self, 
                 layers, 
                 activation=torch.nn.Tanh, 
                 last_layer_activation=None, 
                 initialization=None,
                 m=1,
                 freq_stds=None):
        super(FourierEmbeddedDNN, self).__init__()
        # fourier embedding is applied prior to passing into neural net, 
        # need to make sure dimensions match
        assert layers[0] == 2*m
        # build main DNN
        self.layer_spec = layers
        self.layers = self.build_nn(
            layers, activation, last_layer_activation, initialization
        )
        # build fourier feature embedding
        self.fourier_embedding = self.build_embedding(m, freq_stds)
        
        # build final aggregator to combine outputs of different scale fourier embeddings
        self.build_aggregator()
    
    def build_nn(self, layers, activation, last_layer_activation, initialization):
        self.depth = len(layers) - 1
        # set up layer order dict
        self.activation = activation
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        if last_layer_activation is not None:
            layer_list.append(
            ('activation_%d' % (self.depth - 1), last_layer_activation())
        )

        layerDict = OrderedDict(layer_list)
        return torch.nn.Sequential(layerDict)
    
    def build_embedding(self, num_freqs, freq_stds):
        # number of feature embeddings correspond to length of standard 
        # deviations specified. If `None`, by default uses only 1 embedding
        # standard Gaussian.
        if freq_stds:
            self.num_embeddings = len(freq_stds)
        else:
            self.num_embeddings = 1
            freq_stds = [1.0]
        # draw frequency matrix
        freq_matrix = [torch.randn(num_freqs, requires_grad=False) for _ in range(self.num_embeddings)]
        for i in range(self.num_embeddings):
            # scale by frequency standard deviation
            freq_matrix[i] = torch.tensor(freq_stds[i])*freq_matrix[i]
        return freq_matrix
    
    def build_aggregator(self):
        # number of fourier embeddings
        k = self.num_embeddings
        # size of hidden layer final outputs
        num_out = self.layer_spec[-1]
        # create trainable aggregating weights for each embedding (simple linear aggregation
        # , may also consider computing another nonlinear activation for each embedding, then 
        # summing all outputs).
        self.aggregator = torch.nn.Linear(num_out*k, 1)
        
    def fourier_lifting(self, x, freq):
        # input x has size (N x 1), output has size (N x 2*m) where m is number of Fourier bases
        
        # has size (N x m)
        x = freq * x
        # lift to sin and cos space
        x = torch.concat(
            [
                torch.cos(2*torch.pi*x), 
                torch.sin(2*torch.pi*x)
            ], dim=1
        )
        return x
    
    def forward(self, x):
        # inputs x has size (N x 1)
        # create Fourier features
        lifted = []
        for i in range(self.num_embeddings):
            lifted.append(self.fourier_lifting(x, self.fourier_embedding[i]))
        # lifted is a length-k list of (N x 2*m) tensors of lifted features according to 
        # k different scales.
        
        # now pass each (N x 2*m) features into the hidden layers
        for i in range(self.num_embeddings):
            lifted[i] = self.layers(lifted[i])
        
        # lifted is a length-k list of (N x num_out) tensor of transformed fourier features
        # now concatenate into (N x num_out*k) and pass into aggregator to obtain (N x 1) prediction
        lifted = torch.concat(lifted, dim=1)
        # final aggregation
        lifted = self.aggregator(lifted)
        return lifted

##########################################################################################
# Fourier-embedded DNN (spatio-temporal dependence)
##########################################################################################
class FourierEmbeddedDNN2d(torch.nn.Module):
    """ 
        A deep neural network suitable for learning space-time functions by stacking
        two separate Fourier embedded nets together and combining the results.

        The spatial and temporal Fourier mappings share the same DNN for hidden transformations.
    """
    def __init__(self, layers, activation=torch.nn.Tanh, 
                 last_layer_activation=None, initialization=None, 
                 m=1, freq_stds=None):
        super(FourierEmbeddedDNN2d, self).__init__()
        # fourier embedding is applied prior to passing into neural net, 
        # need to make sure dimensions match
        assert layers[0] == 2*m
        if freq_stds is not None:
            assert freq_stds.shape[1] == 2, "Specify both space and time frequency scales. "
        else:
            freq_stds = np.ones([1, 2])
        
        # build main DNN
        self.layer_spec = layers
        self.layers = self.build_nn(
            layers, activation, last_layer_activation, initialization
        )
        # build fourier feature embedding
        self.fourier_embedding_time, self.fourier_embedding_space = self.build_embedding(m, freq_stds)
        
        # build final aggregator to combine outputs of different scale fourier embeddings
        self.build_aggregator()
    
    def build_nn(self, layers, activation, last_layer_activation, initialization):
        self.depth = len(layers) - 1
        # set up layer order dict
        self.activation = activation
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        if last_layer_activation is not None:
            layer_list.append(
            ('activation_%d' % (self.depth - 1), last_layer_activation())
        )

        layerDict = OrderedDict(layer_list)
        return torch.nn.Sequential(layerDict)

    def build_embedding(self, num_freqs, freq_stds):
        # number of feature embeddings correspond to length of standard 
        # deviations specified. If `None`, by default uses only 1 embedding
        # standard Gaussian.
        self.num_embeddings = freq_stds.shape[0]
        # time domain scales
        freq_stds_t = freq_stds[:, 0].flatten()
        # spatial domain scales
        freq_stds_x = freq_stds[:, 1].flatten()
        
        # draw frequency matrix (time)
        freq_matrix_t = [torch.randn(num_freqs, requires_grad=False) for _ in range(self.num_embeddings)]
        # draw frequency matrix (space)
        freq_matrix_x = [torch.randn(num_freqs, requires_grad=False) for _ in range(self.num_embeddings)]
        
        for i in range(self.num_embeddings):
            # scale by frequency standard deviation
            freq_matrix_t[i] = torch.tensor(freq_stds_t[i])*freq_matrix_t[i]
            freq_matrix_x[i] = torch.tensor(freq_stds_x[i])*freq_matrix_x[i]
        return freq_matrix_t, freq_matrix_x

    def fourier_lifting(self, x, freq):
        # input x has size (N x 1), output has size (N x 2*m) where m is number of Fourier bases
        
        # has size (N x m)
        x = freq * x
        # lift to sin and cos space
        x = torch.concat(
            [
                torch.cos(2*torch.pi*x), 
                torch.sin(2*torch.pi*x)
            ], dim=1
        )
        return x
    
    def build_aggregator(self):
        # number of fourier embeddings
        k = self.num_embeddings
        # size of hidden layer final outputs
        num_out = self.layer_spec[-1]
        self.aggregator = torch.nn.Linear(num_out*k, 1)
    
    def forward(self, inputs):
        # inputs has size (N x 2)
        # create Fourier features for t
        t = inputs[:, 0][:, None]
        x = inputs[:, 1][:, None]
        
        lifted_t, lifted_x = [], []
        for i in range(self.num_embeddings):
            lifted_t.append(self.fourier_lifting(t, self.fourier_embedding_time[i]))
            lifted_x.append(self.fourier_lifting(x, self.fourier_embedding_space[i]))
            
        # lifted is a length-k list of (N x 2*m) tensors of lifted features according to 
        # k different scales.
        
        # now pass each (N x 2*m) features into the hidden layers
        for i in range(self.num_embeddings):
            lifted_t[i] = self.layers(lifted_t[i])
            lifted_x[i] = self.layers(lifted_x[i])
        
        
        # lifted is a length-k list of (N x num_out) tensor of transformed fourier features
        # now concatenate into (N x num_out*k) and pass into aggregator to obtain (N x 1) prediction
        lifted_t = torch.concat(lifted_t, dim=1)
        lifted_x = torch.concat(lifted_x, dim=1)
        
        # elementwise multiplication
        lifted = lifted_t * lifted_x
        # final aggregation
        lifted = self.aggregator(lifted)
        return lifted














