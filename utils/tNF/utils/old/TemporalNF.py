import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.distributions.transforms as transform
import matplotlib.pyplot as plt

class DNN(nn.Module):
    """ Basic fully connected NN. """
    def __init__(self, in_dim, out_dim, hidden_dim, nlayers, activation=torch.nn.Tanh):
        super().__init__()
        net = nn.ModuleList()
        for l in range(nlayers):
            # linear weights
            net.append(
                nn.Linear(l==0 and in_dim or hidden_dim, l==nlayers-1 and out_dim or hidden_dim)
            )
            # activation
            net.append(activation())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Flow(transform.Transform, nn.Module):
    """ Overarching class for all flow models. """
    def __init__(self):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
    
    def init_parameters(self):
        """ Initialize all trainable parameters (declared using 
        torch.nn.Parameter(). Provides different modes of initialization
        if needed.  """
        for param in self.parameters():
            # use random parameters
            param.data.uniform_(0.1, 0.1)

            
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)
    
    # forward evaluation: x = f(z)
    def forward(self, z):
        pass


class tAutoregressive(Flow):
    """ 
        Dinh et. al (2020) RealNVP architecture, with
        time dependence. The time dependence is assumed
        to be p(t, x)
    """
    def __init__(
        self, 
        space_dim, 
        hidden_dim=32, 
        n_layers=3, 
        activation=nn.Sigmoid
    ):
        super(tAutoregressive, self).__init__()
        D = 1 + space_dim
        # half of spatial dimension + time
        self.k = 1 + space_dim // 2
        # shift
        self.g_mu = DNN(self.k, D-self.k, hidden_dim, n_layers, activation)
        # scale
        self.g_sig = DNN(self.k, D-self.k, hidden_dim, n_layers, activation)
        self.init_parameters()
        self.bijective = True

    def _call(self, z):
        # z_k => (t, x_1:d/2);  z_D => (x_d/2:D)
        z_k, z_D = z[:, :self.k], z[:, self.k:]
        # apply scale and shift only on z_k
        zp_D = z_D * torch.exp(self.g_sig(z_k)) + self.g_mu(z_k)
        return torch.cat((z_k, zp_D), dim = 1)
    
    def _inverse(self, z):
        # zp_k => (t, y_1:d/2);  zp_D => (y_d/2:D)
        zp_k, zp_D = z[:, :self.k], z[:, self.k:]
        z_D = (zp_D - self.g_mu(zp_k)) / torch.exp(self.g_sig(zp_k))
        return torch.cat((zp_k, z_D), dim=1)
    
    def log_abs_det_jacobian(self, z):
        """ Same as the jacobian without time dependence. """
        z_k = z[:, :self.k]
        # transformed variables
        g_k = self.g_sig(z_k)
        if z_k.shape[1] > 1:
            return (g_k).sum(1).reshape(-1, 1)
        return (g_k).sum(1).reshape(-1, 1)

class tReverseFlow(Flow):
    """ 
        Flow map that implements a permutation. The
        time dimension is never permuted.
    """
    def __init__(self, space_dim):
        super(tReverseFlow, self).__init__()
        # define reverse permutation (in space)
        self.permute = torch.arange(space_dim-1, 0, -1)
        # add time dimension
        self.permute = torch.cat((torch.tensor([0]), self.permute), dim=0)
        # define inverse permutation (in space)
        self.inverse = torch.argsort(self.permute)
    
    def _call(self, z):
        return z[:, self.permute]

    def _inverse(self, z):
        return z[:, self.inverse]
    
    def log_abs_det_jacobian(self, z):
        """ Jacobian will always be 1, log jac is 0. """
        z_space = z[:, 1:]
        return torch.zeros(z_space.shape[0], 1)
    
class tShuffleFlow(tReverseFlow):
    """ Random permutation in space. """
    def __init__(self, space_dim):
        super(tShuffleFlow, self).__init__(space_dim)
        self.permute = torch.randperm(space_dim)+1
        # add time dimension
        self.permute = torch.cat((torch.tensor([0]), self.permute), dim=0)
        
        self.inverse = torch.argsort(self.permute)

class tSwapFlow(tReverseFlow):
    """ Swaps half of the dimensions in space, assuming the number of 
    dimensions is even. """
    def __init__(self, space_dim):
        assert space_dim % 2 == 0
        super(tSwapFlow, self).__init__(space_dim)
        self.permute = torch.arange(1, 1+space_dim)
        tmp = space_dim // 2
        # swap half the the indices with the other half (i.e. [1, 2, 3, 4]=>[3, 4, 1, 2])
        tmp1, tmp2 = self.permute[:tmp].clone(), self.permute[tmp:].clone()
        self.permute[:tmp] = tmp2
        self.permute[tmp:] = tmp1
        # add time dimension (never permuted)
        self.permute = torch.cat((torch.tensor([0]), self.permute), dim=0)
        self.inverse = torch.argsort(self.permute)
        
        

class tBatchNormFlow(Flow):
    """ Batch normalization only applied on the spatial dimensions. """
    def __init__(self, space_dim, momentum=0.95, eps=1e-5):
        super(tBatchNormFlow, self).__init__()
        # running batch statistics
        self.r_mean = torch.zeros(space_dim)
        self.r_var = torch.ones(space_dim)
        # initialize momentum
        self.momentum = momentum
        self.eps = eps
        # trainable scale and shift
        self.gamma = nn.Parameter(torch.ones(space_dim))
        self.beta = nn.Parameter(torch.zeros(space_dim))
    
    def _call(self, z):
        z_space = z[:, 1:]
        if self.training:
            # Current batch stats
            self.b_mean = z_space.mean(0)
            self.b_var = (z_space - self.b_mean).pow(2).mean(0) + self.eps
            # Running mean and var
            self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        # normalize space
        x_space = (z_space - mean) / var.sqrt()
        # scale and shift back to original scale
        y_space = self.gamma * x_space + self.beta
        return torch.cat((z[:, 0].reshape(-1, 1), y_space), dim=1)
    
    def _inverse(self, x):
        x_space = x[:, 1:]
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        z_space = (x_space - self.beta) / self.gamma
        y_space = z_space * var.sqrt() + mean
        return torch.cat((x[:, 0].reshape(-1, 1), y_space), dim=1)
    
    def log_abs_det_jacobian(self, z):
        """ Same as the jacobian det without time. """
        z_space = z[:, 1:]
        mean = z_space.mean(0)
        var = (z_space - mean).pow(2).mean(0) + self.eps
        # constant in `z`
        log_det = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
        log_det = torch.ones(z_space.shape[0], 1)*log_det
        return log_det.sum(1).reshape(-1, 1)

    
class tNormalizingFlow(nn.Module):
    """ Main architecture for stacking normalizing flow layers. """
    def __init__(self, space_dim, blocks, flow_length):
        super().__init__()
        biject = []
        for f in range(flow_length):
            for flow in blocks:
                biject.append(flow(space_dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # applies series of flows
        for b in range(len(self.bijectors)):
            jac = self.bijectors[b].log_abs_det_jacobian(z).reshape(-1, 1)
            self.log_det.append(jac)
            z = self.bijectors[b](z)
        return z, self.log_det

    def sample(self, z):
        """ 
            Applies inverse maps to convert latent samples into observations.
        """
        # applies series of inverse flows
        n = len(self.bijectors)
        for b in range(n-1, -1, -1):
            z = self.bijectors[b]._inverse(z)
        return z