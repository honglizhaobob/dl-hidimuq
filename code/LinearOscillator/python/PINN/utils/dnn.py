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
        self.depth = len(layers)
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
    def __init__(
        self, 
        num_hidden,
        layer_size, 
        size_out,
        activation=torch.nn.Tanh, 
        last_layer_activation=None, 
        initialization=None, 
        m=1, 
        freq_stds={"space": [1.0], "time": [1.0]}
    ):
        """
            num_hidden (int)                    Number of hidden layers for the network shared by 
                                                all embeddings.
            
            layer_size (int)                    Size of hidden layers.

            size_out (int)                      Size of output multiscale features.

            freq_stds (dict)                    Python dictionary containing fields "time", "space",
                                                where 1d arrays are stored to indicate the standard 
                                                deviations. The length of the arrays are assumed to 
                                                be the number of embeddings for time / space. 

                                                All embeddings are passed through the same deep neural net
                                                with input size 2*m.

                                                By default, random frequencies have unit standard deviation.
        """

        super(FourierEmbeddedDNN2d, self).__init__()
        # fourier embedding is applied prior to passing into neural net
        self.multiscale_feature_size = size_out
        self.layer_spec = [2*m] + [layer_size] * num_hidden + [self.multiscale_feature_size]
        # build main DNN
        self.layers = build_nn(layers=self.layer_spec, activation=activation, last_layer_activation=last_layer_activation)
  
        # build fourier feature embedding
        self.time_freqs, self.space_freqs = self.random_freqs(m, freq_stds)
        self.fourier_embedding_time, self.fourier_embedding_space = self.random_freqs(m, freq_stds)
        
        # build final aggregator to combine outputs of different scale fourier embeddings
        self.aggregator = self.build_aggregator()

    def random_freqs(self, m, freq_stds):
        """
            Assume input dimension is 1-dimensional, draws Gaussian random vectors representing
            frequency levels for the embedding. All embeddings have the same size `m`.
        """
        # unpack standard deviation specifications
        freq_stds_t = torch.tensor(freq_stds["time"], requires_grad=False)
        freq_stds_x = torch.tensor(freq_stds["space"], requires_grad=False)
        num_embeddings_t = len(freq_stds_t)
        num_embeddings_x = len(freq_stds_x)
        # frequency matrices of size (m x num_embeddings), each row scaled properly
        freq_mat_x = torch.randn(m, num_embeddings_x, requires_grad=False)
        freq_mat_t = torch.randn(m, num_embeddings_t, requires_grad=False)
        # scale to proper frequencies
        space_scale = torch.diag(freq_stds_x)
        time_scale = torch.diag(freq_stds_t)
        freq_mat_x = freq_mat_x @ space_scale
        freq_mat_t = freq_mat_t @ time_scale
        return freq_mat_t, freq_mat_x

    def fourier_lifting(self, x, freq):
        """
            freq (array)                       length-m vector indicating preferred level of Fourier 
                                               frequency to be learned.
        """
        # input x has size (n x 1), output has size (n x 2*m) where m is number of Fourier bases
        x = x.reshape(-1, 1)
        freq = freq.flatten()
        
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
        """ 
            Due to Hadamard products for time and space, the size of the aggregator will be 
            (num_out * num_embeddings_t * num_embeddings_x) mapping to 1
        """
        num_out = self.multiscale_feature_size
        m, num_embeddings_t = self.fourier_embedding_time.shape
        m, num_embeddings_x = self.fourier_embedding_space.shape
        # size of final network
        k = num_out * num_embeddings_t * num_embeddings_x
        return torch.nn.Linear(k, 1)
    
    def forward(self, inputs):
        """ 
            Inputs should be of size (n x 2) 
            Outputs should be of size (n x 1)
        """
        n = inputs.shape[0]
        assert inputs.shape[1] == 2
        k = self.multiscale_feature_size

        # separate inputs (n x 1)
        t = inputs[:, 0][:, None]
        x = inputs[:, 1][:, None]

        # number of embeddings in each dimension
        m, num_embeddings_t = self.fourier_embedding_time.shape
        _, num_embeddings_x = self.fourier_embedding_space.shape
        # compute feature liftings for each dimension
        lifted_t, lifted_x = torch.zeros(num_embeddings_t, n, 2*m), torch.zeros(num_embeddings_x, n, 2*m)
        # lift t
        for i in range(num_embeddings_t):
            lifted_t[i, :, :] = self.fourier_lifting(t, self.fourier_embedding_time[:, i])
        # lift x
        for i in range(num_embeddings_x):
            lifted_x[i, :, :] = self.fourier_lifting(x, self.fourier_embedding_space[:, i])
        # pass through deep neural network (num_embeddings x n x 2m) => (num_embeddings x n x k)
        lifted_t, lifted_x = self.layers(lifted_t), self.layers(lifted_x)

        # form hadamard product (num_embedding_t*num_embedding_x x n x k)
        lifted = []
        for i in range(num_embeddings_t):
            for j in range(num_embeddings_x):
                lifted.append(torch.mul(lifted_t[i, :, :], lifted_x[j, :, :]))
        lifted = torch.stack(lifted)
        # permute dimensions and combine last two dimensions (n x num_embedding_x*num_embedding_t*k)
        lifted = lifted.permute(1, 0, 2).reshape(n, -1)
        # final aggregation
        out = self.aggregator(lifted)
        return out

##########################################################################################
# Fourier-embedded DNN (spatio-temporal dependence), old implementation
##########################################################################################
class FourierEmbeddedDNN2dOld(torch.nn.Module):
    """ 
        Old implementation: A deep neural network suitable for learning space-time functions by stacking
        two separate Fourier embedded features together and combining the results.
        The spatial and temporal Fourier mappings share the same DNN for hidden transformations.
    """
    def __init__(self, layers, activation=torch.nn.Tanh, 
                 last_layer_activation=None, initialization=None, 
                 m=1, freq_stds=None):
        super(FourierEmbeddedDNN2dOld, self).__init__()
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

##########################################################################################
# Fourier-embedded DNN (spatio-temporal dependence) Version 2
##########################################################################################
class FourierProductEmbeddedDNN2d(torch.nn.Module):
    """
        Alternative implementation of the spatio-temporal Fourier embedded deep neural net.
        The difference of this architecture from `FourierEmbeddedDNN2d` is that we use 
        separate deep neural networks to lift the spatial and temporal dimensions.

        This architecture products users the option of whether or not to transform
        time or spatial dimension. 
    """

####################################################################################################
# Helper functions for neural nets
####################################################################################################
def build_nn(layers, activation, last_layer_activation=None):
    depth = len(layers)-2
    # set up layer order dict
    layer_list = list()
    for i in range(depth - 1): 
        layer_list.append(
            ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
        )
        layer_list.append(('activation_%d' % i, activation()))
        
    layer_list.append(
        ('layer_%d' % (depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
    )
    if last_layer_activation is not None:
        layer_list.append(
        ('activation_%d' % (depth - 1), last_layer_activation())
    )

    layerDict = OrderedDict(layer_list)
    return torch.nn.Sequential(layerDict)










