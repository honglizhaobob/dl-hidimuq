# This is an experimental implementation of the hierarchical PINNs, for inverse problems.
# The main idea is to decompose the neural network solution to the PDE as a sum of 
# neural networks:
#
#           u(t, x) = torch.sum([v1(t, x), v2(t, x), ..., v_m(t, x)])    (1)
# where each v_i (1 <= i <= m) focuses on learning the PDE solution at a 
# specific frequency. The underlying PDE is assumed to be:
#
#           p_t + d/dx ( V(t, x) * p ) = 0
# where V(t, x) exhibits high-frequency behavior in time. 
#
# The neural nets v_i may have varying sizes and Fourier feature encoding, and 
# are trained by levels. At each level, a prediction is made by freezing 
# other levels and aggregating the solutions according to (1). The result is 
# then back-propagated from a low-fidelity kernel density estimate. 

# Please refer to `PhysicsInformedROPDF.py` for a description of the input-output 
# ordering of the neural networks. 



################################################################################
# List of imports
################################################################################
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

# import neural nets
from .utils.dnn import *
from .utils.helpers import *

######################################################################
class G_Net(FourierProductEmbeddedDNN2d):
    """ 
        A spatio-temporal DNN with Fourier feature mappings in each dimension.
    """
    def __init__(self, **kwargs):
        super(G_Net, self).__init__(**kwargs)

######################################################################
# PINN
######################################################################
class HierarchicalPhysicsInformedROPDF(nn.Module):
    def __init__(
        self, coeff_net_spec
    ):
        super(HierarchicalPhysicsInformedROPDF, self).__init__()
        # start with an empty neural net group
        self.model_group = []
        # number of levels
        self.num_levels = 0
        # build coeffient neural net
        self.r_model = self.build_coefficient_net(coeff_net_spec)
        # build model aggregation
        self.aggregator = None
    
    def add_model(self, model):
        # adds a neural net into the model group for ensemble prediction
        self.model_group.append(model)
        # update number of levels
        self.num_levels = len(self.model_group)
    
    def build_coefficient_net(self, spec):
        pass

    def build_aggregator(self):
        if self.num_levels == 0:
            raise ValueError("No dimensions to aggregate, add models first. ")
        return torch.nn.Linear(self.num_levels, 1)

    def freeze(self, net):
        # stops a net from accumulating gradients
        # https://stackoverflow.com/questions/68377722/in-pytorch-model-training-how-to-freeze-unfreeze-and-freeze-again-some-params
        # https://blog.csdn.net/m0_46653437/article/details/112651078
        net.requires_grad_(False)

    def unfreeze(self, net):
        # enable training again for the input net
        net.requires_grad_(True)

    
    def forward(self, inputs):
        if self.num_levels == 0:
            raise ValueError()
        if self.aggregator is None:
            raise ValueError("Aggregator not initialized. ")
        # linear combination of all levels
        n = inputs.shape[0]
        assert inputs.shape[1] == 2
        assert len(self.model_group) == self.num_levels
        res = torch.zeros(n, self.num_levels)
        for i in range(self.num_levels):
            res[:, i] = self.model_group[i](inputs)
        # pass through aggregator
        res = self.aggregator(res)
        return res


    # level dependent loss functions (requires freeze and unfreeze)
    def physics_loss(self, inputs, level):
        pass

    def data_loss(self, inputs, level):
        pass

    # train this model
    def train(self, X, y, loss, optim, scheduler):
        """
            Loss function is an input as it changes by level, furthermore,
            `optim` and `scheduler` options are provided in case one would 
            like to use different optimizers for each level. 
        """
        pass




        