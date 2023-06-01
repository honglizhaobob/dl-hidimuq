# Vanilla normalizing flows, no time dependence assumed.
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.init as init
import torch.distributions.transforms as transform
import torch.nn.functional as functional
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
from sklearn import datasets

class DNN(nn.Module):
    """ Basic fully connected NN. """
    def __init__(self, 
        in_dim, out_dim, 
        hidden_dim, 
        nlayers, 
        activation=torch.nn.ReLU,
        activate_last=True
    ):
        super().__init__()
        net = nn.ModuleList()
        for l in range(nlayers):
            # linear weights
            net.append(
                nn.Linear(l==0 and in_dim or hidden_dim, l==nlayers-1 and out_dim or hidden_dim)
            )
            if l == nlayers-1 and not activate_last:
                # do not activate the last layer
                break
            # activation
            net.append(activation())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class AffineCouplingFlow(nn.Module):
    """ 
        Building block for RealNVP, not suitable for 1d distributions.
        ~
        Reference: https://arxiv.org/abs/1605.08803. 
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, activation, mask):
        super(AffineCouplingFlow, self).__init__()
        # fix masking tensor 
        self.mask = nn.Parameter(mask, requires_grad=False)
        # scale NN
        self.g_sig = DNN(in_dim, out_dim, hidden_dim, n_layers, activation)
        # shift NN
        self.g_mu = DNN(in_dim, out_dim, hidden_dim, n_layers, activation, activate_last=False)
        # additional scaling for scale NN
        self.g_sig_scale = nn.Parameter(torch.Tensor(in_dim))
        init.normal_(self.g_sig_scale)
    
    def forward(self, z):
        """ 
            generative direction x = f(z)
        """
        s = self.g_sig_scale*self.g_sig(z*self.mask)
        t = self.g_mu(z*self.mask)
        # apply masking
        x = self.mask*z + (1-self.mask)*(torch.exp(s)*z + t)
        logabsdet = torch.sum((1-self.mask)*s, -1)
        return x, logabsdet
    
    def inverse(self, x):
        """
            normalizing direction z = f^-1(x)
        """
        s = self.g_sig_scale*self.g_sig(x*self.mask)
        t = self.g_mu(x*self.mask)
        z = self.mask*x + (1-self.mask)*(torch.exp(-s)*(x-t))
        logabsdet = torch.sum((1-self.mask)*(-s), -1)
        return z, logabsdet

class VanillaNormFlow(nn.Module):
    """ 
        A normalizing flow layer that maps the outputs 
        of the last layer to a small interval using the 
        Tanh function. Default [-4, 4]. Used to prevent 
        instability during training.
    """
    def __init__(self, in_dim, out_dim, scaling=4.0):
        super(VanillaNormFlow, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scaling
    
    def inverse(self, z):
        """
            Generative direction x = f(z)
        """
        x = 0.5*(torch.log(1+z/self.scale)-torch.log(1-z/self.scale))
        logabsdet = torch.sum(
            torch.log(
                torch.abs(
                    (1/self.scale)*1/(1-(z/self.scale)**2)
                )
            ), 
        -1)
        return x, logabsdet

    def forward(self, x):
        """
            Normalizing direction z = f^-1(x), maps to [-scale, scale]
            to stablize during training.
        """
        tmp = torch.tanh(x)
        z = self.scale * tmp
        logabsdet = torch.sum(
            torch.log(
                torch.abs(
                    self.scale * (
                        1-tmp**2
                    )
                )
            )
        , -1)
        return z, logabsdet
    


class tBatchNormFlow(nn.Module):
    """ Batch normalization only applied on the spatial dimensions. """
    def __init__(self, space_dim, momentum=0.95, eps=1e-5):
        super(tBatchNormFlow, self).__init__()
        self.space_dim = space_dim
        # batch statistics
        self.b_mean = torch.zeros(space_dim)
        self.b_var = torch.ones(space_dim)
        # running batch statistics
        self.r_mean = torch.zeros(space_dim)
        self.r_var = torch.ones(space_dim)
        # initialize momentum
        self.momentum = momentum
        self.eps = eps
        # trainable scale and shift
        self.gamma = nn.Parameter(torch.ones(space_dim))
        self.beta = nn.Parameter(torch.zeros(space_dim))
    
    def forward(self, z):
        """
            Generative direction: x = f(z)
        """
        batch_size = z.shape[0]
        z_space = z[:, 1:]
        #z_space = z
        if self.training:
            mean = self.b_mean
            var = self.b_var
        else:
            mean = self.r_mean
            var = self.r_var
        x_space = (z_space - self.beta) / self.gamma
        x_space = x_space * var.sqrt() + mean
        x = torch.cat((z[:, 0].reshape(-1, 1), x_space), dim=1)
        #x = x_space
        # compute log-likelihood: det(a*I) = a^n
        logabsdet = (0.5*torch.log(var)-torch.log(self.gamma)).sum().repeat(batch_size)

        return x, logabsdet

    def inverse(self, x):
        """ 
            Stablize during training by dividing by batch statistics.

            Normalizing direction: z = f^-1(x)
        """
        x_space = x[:, 1:]
        #x_space = x
        batch_size = x.shape[0]
        if self.training:
            # current batch stats
            self.b_mean = x_space.mean(0)
            self.b_var = (x_space - self.b_mean).pow(2).mean(0) + self.eps

            # save for current layer transformation
            mean = self.b_mean
            var = self.b_var

            # update running mean and var
            self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
            self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
            
        else:
            mean = self.r_mean
            var = self.r_var
        # normalize space
        z_space = (x_space - mean) / var.sqrt()
        # scale and shift back to original scale
        z_space = self.gamma * z_space + self.beta

        # compute log-likelihood
        logabsdet = (torch.log(self.gamma)-0.5*torch.log(var)).sum().repeat(batch_size)
        z = torch.cat((x[:, 0].reshape(-1, 1), z_space), dim=1)
        #z = z_space
        return z, logabsdet


class tAffineCouplingFlow(nn.Module):
    """
        Time dependent affine coupling blocks.
        The time dimension is not transformed.

        Masking is also only done in space. The 
        first dimension of the flows are assumed to
        be time.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, activation, mask):
        super(tAffineCouplingFlow, self).__init__()
        # fix masking tensor 
        self.mask = nn.Parameter(mask, requires_grad=False)
        # scale NN
        self.g_sig = DNN(1+in_dim, 1+out_dim, hidden_dim, n_layers, activation)
        # shift NN
        self.g_mu = DNN(1+in_dim, 1+out_dim, hidden_dim, n_layers, activation, activate_last=False)
        # additional scaling for scale NN
        self.g_sig_scale = nn.Parameter(torch.Tensor(in_dim+1), requires_grad=True)
        init.normal_(self.g_sig_scale)
        

    def forward(self, z):
        """ 
            generative direction x = f(z)
        """
        s = self.g_sig_scale*self.g_sig(z*self.mask)
        t = self.g_mu(z*self.mask)
        # apply masking
        x = self.mask*z + (1-self.mask)*(torch.exp(s)*z + t)
        logabsdet = torch.sum((1-self.mask)*s, -1)
        return x, logabsdet
    
    def inverse(self, x):
        """
            normalizing direction z = f^-1(x)
        """
        s = self.g_sig_scale*self.g_sig(x*self.mask)
        t = self.g_mu(x*self.mask)
        z = self.mask*x + (1-self.mask)*(torch.exp(-s)*(x-t))
        logabsdet = torch.sum((1-self.mask)*(-s), -1)
        return z, logabsdet

class tVanillaNormFlow(nn.Module):
    """ 
        A normalizing flow layer that maps the outputs 
        of the last layer to a small interval using the 
        Tanh function. Default [-4, 4]. Used to prevent 
        instability during training.

        Added (dummy) time dimension, which is not transformed.
    """
    def __init__(self, in_dim, out_dim, scaling=10.0):
        super(tVanillaNormFlow, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scaling

    def forward(self, z):
        """
            Generative direction x = f(z)
        """
        tmp = torch.tanh(z[:, 1:])
        x = torch.cat([z[:, 0].reshape(-1, 1), self.scale * tmp], 1)
        logabsdet = torch.sum(
            torch.log(
                torch.abs(
                    self.scale * (
                        1-tmp**2
                    )
                )
            )
        , -1)
        return x, logabsdet
    
    def inverse(self, x):
        """
            Normalizing direction z = f^-1(x)
        """
        tmp = x[:, 1:]
        z2 = 0.5*(torch.log(1+tmp/self.scale)-torch.log(1-tmp/self.scale))
        z = torch.cat([x[:, 0].reshape(-1, 1), z2], 1)
        logabsdet = torch.sum(
            torch.log(
                torch.abs(
                    (1/self.scale)*1/(1-(tmp/self.scale)**2)
                )
            ), 
        -1)
        return z, logabsdet


######################################################################
class ResidualFlow(nn.Module):
    """ 
        Residual networks normalizing flow layer, made invertible
        by fixed point method. In this implementation, we parameterize
        the inverse map f^-1(x), by a ResNet.

        References:
            https://arxiv.org/abs/1811.00995
            https://arxiv.org/abs/1906.02735
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, activation=nn.ReLU):
        super(ResidualFlow, self).__init__()
        # create neural net
        
        

# https://github.com/xqding/RealNVP/blob/master/Real%20NVP%20Tutorial.ipynb


######################################################################
class NormalizingFlow(nn.Module):
    """ Main architecture for stacking normalizing flow layers. """
    def __init__(self, blocks, flow_length):
        super(NormalizingFlow, self).__init__()
        biject = []
        for f in range(flow_length):
            for flow in blocks:
                biject.append(flow)
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.k = len(self.bijectors)

    def forward(self, z):
        """ 
            x = f(z) = f_K ( f_K-1( ... f_1 ( f_0(z))))
        """
        log_det = []
        # applies series of flows
        for b in range(self.k):
            # compute generative direction of one block
            tmp, logabsdet = self.bijectors[b](z)
            # store jac
            log_det.append(logabsdet)
            z = tmp
        x = z
        # sum all log abs determinants
        sum_logabsdet = sum(log_det)
        return x, sum_logabsdet
    
    def inverse(self, x):
        """
            z = f^-1(x) = f_0^-1( f_1^-1 ( ... ( f_K^-1(x))))
        """
        log_det = []
        # applies series of inverse flows
        for b in range(self.k-1, -1, -1):
            # compute normalizing direection of one block
            tmp, logabsdet = self.bijectors[b].inverse(x)
            # store jac
            log_det.append(logabsdet)
            x = tmp
        z = x
        # sum all log abs dets
        sum_logabsdet = sum(log_det)
        return z, sum_logabsdet



