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
        self, indim, outdim, data_path,
        scheduler=None, optimizer="adam"
    ):
        super(HierarchicalPhysicsInformedROPDF, self).__init__()
        