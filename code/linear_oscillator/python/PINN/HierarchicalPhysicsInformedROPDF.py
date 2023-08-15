# This is an experimental implementation of the hierarchical PINNs, for inverse problems.
# The main idea is to decompose the neural network solution to the PDE as a sum of 
# neural networks:
#
#           u(t, x) = torch.sum([v1(t, x), v2(t, x), ..., v_m(t, x)])    (1)
# where each v_i (1 <= i <= m) focuses on learning the PDE solution at a 
# specific frequency. The underlying PDE is assumed to be:
#
#           p_t + d/dx ( V(t, x) * p ) = d^2/dx^2 ( D(t, x) * p )
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
# PINN
######################################################################
class HierarchicalPhysicsInformedROPDF(nn.Module):
    def __init__(
        self, advection_net, diffusion_net
    ):
        """ 
            Advection and diffusion neural networks are created 
            outside the physics-informed architecture.
        """
        super(HierarchicalPhysicsInformedROPDF, self).__init__()
        # start with an empty neural net group
        self.model_group = []
        # number of levels
        self.num_levels = 0
        # build model aggregation
        self.aggregator = None
        # build advection coefficient net
        self.G_nn = advection_net
        # build diffusion coefficient net
        self.D_nn = diffusion_net
    
    def add_model(self, model):
        # adds a neural net into the model group for ensemble prediction
        self.model_group.append(model)
        # update number of levels
        self.num_levels = len(self.model_group)
    
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

    def freeze_all(self):
        """ Helper function to freeze all models. """
        assert self.num_levels != 0
        for i in range(self.num_levels):
            self.freeze(self.model_group[i])

    def unfreeze_all(self):
        """ Helper function to unfreeze all models. """
        assert self.num_levels != 0
        for i in range(self.num_levels):
            self.unfreeze(self.model_group[i])

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
        #res = self.aggregator(res)
        # currently, aggregator is a simple sum over all levels
        res = torch.sum(res, dim=1)
        return res

    # level dependent loss functions (requires freeze and unfreeze)
    def physics_loss(self, inputs, level):
        """ 
            PDE loss function at level k. Let all levels be 
            l=1, 2, ..., k-1, k, k+1, ..., L, this function 
            returns a loss function that only depends on 
            level k's parameters.
        """
        assert 0 <= level < self.num_levels
        # define loss function at level k
        residual = self.pde_eval(inputs, level)
        loss = torch.mean(residual ** 2)
        return loss

    def pde_eval(self, inputs, k):
        """ 
            Evaluates the underlying PDE operator, helper
            function for physics loss.

            The PDE op. is assumed to be conservative form
            advection-diffusion.
        """
        n = inputs.shape[0]
        # separate inputs into time and spatial variables assuming order (t, x)
        inputs_t = inputs[:, 0][:, None]
        inputs_x = inputs[:, 1][:, None]

        # evaluate NN predictions up to level k
        p = torch.zeros(n, 1, requires_grad=True)
        for i in range(0, k+1):
            p[:, 0] += self.model_group[i](inputs)
        # apply linear PDE operator

        # time derivative and spatial derivative
        deriv = self.gradient(
            p, inputs, order=1
        )
        p_dt = deriv[:, 0][:, None]
        p_dx = deriv[:, 1][:, None]
        
        # predict coefficients
        g_eval = self.G_nn(inputs)
        d_eval = self.D_nn(inputs)

        # evaluate advection-diffusion dynamics
        # p_t + d/dx(G(t, x) * p) - (d/dx)^2(D(t, x)*p) = 0
        tmp = g_eval * p
        dGpdx = self.gradient(
            tmp, inputs, order=1
        )[:, 1][:, None]

        # d/dx D(t, x)
        D_dx = self.gradient(
            d_eval, inputs, order=1
        )[:, 1][:, None]
        # d/dx(D * p) = D_dx * p + D * p_dx
        tmp = D_dx * p + d_eval * p_dx
        dDpdxx = self.gradient(
            tmp, inputs, order=1
        )[:, 1][:, None]

        # assemble residual
        residual = (p_t + dGpdx - dDpdxx)
        return residual

    def data_loss(self, inputs, p_exact, level):
        """ 
            Data loss at level k is computed by freezing all other levels 
            and making an aggregated prediction, such that the loss only 
            depends on parameters at level k.
        """
        n = inputs.shape[0]
        # separate inputs into time and spatial variables assuming order (t, x)
        inputs_t = inputs[:, 0][:, None]
        inputs_x = inputs[:, 1][:, None]
        assert 0 <= level < self.num_levels

        # evaluate all neural nets, but only level k is back-propped
        p_pred = torch.zeros(n, 1, requires_grad=True)
        for i in range(self.num_levels):
            p_pred[:, 0] += self.model_group[i](inputs)
        
        residual = p_pred - p_exact
        loss = torch.mean(residual ** 2)
        return loss

    def prepare_level(self, level):
        """ Prepare the weights in this level by freezing all but `level`. """
        assert self.num_levels != 0
        self.freeze_all()
        self.unfreeze(self.model_group[level])
        

    # train this model at all levels
    def train(self, X, y, level, batch_size, epochs, optim, scheduler, shuffle=True):
        """
            Loss function is an input as it changes by level, furthermore,
            `optim` and `scheduler` options are provided in case one would 
            like to use different optimizers for each level. 

            Trains level-k loss function in batches with `batch_size`, the 
            PDE loss is randomly sampled from the interior domain with the 
            same batch size as the data during each iteration.
        """
        n = X.shape[0]
        assert X.shape[1] == 2
        assert len(y) == n

        # prepare weights for this level (freeze all other levels)
        self.prepare_level(level=level)

        # determine number of batches
        num_batches = int(n / batch_size)
        all_epoch_pde_loss = []
        all_epoch_data_loss = []

        for i in range(epochs):
            print("------------------------------------------------------------------\n")
            print("|                  Level {}, Epoch {}                            |\n".format(level+1, i+1))
            print("------------------------------------------------------------------\n")
            self.train(True)
            if shuffle:
                # generate permutation indices
                tmp = np.random.permutation(N)
                # the `.data` detaches computational graph
                X, y = X[tmp, :].data.clone(), y[tmp, :].data.clone()

            # train this loss
            # loop over batches
            # ----------------------------------------
            # Define batch-wise variables
            #
            all_batch_losses_data = []
            all_batch_losses_pde = []
            # ----------------------------------------
            for idx in range(num_batches):
                if idx % batch_print == 0:
                    print("| => | Batch {} |\n".format(idx+1))
                # ----------------------------------------
                #
                #           Closure definition
                #
                # ----------------------------------------
                # define closure for backpropagation
                def closure():
                    # zero out gradients
                    optim.zero_grad()
                    # get batch data
                    start_idx = idx*batch_size
                    end_idx = (idx+1)*batch_size
                    if idx + 1 == num_batches:
                        # if last batch
                        end_idx = -1
                    Xb, yb = X[start_idx:end_idx, :].data.clone(), y[start_idx:end_idx, :].data.clone()
                    Xb.requires_grad(True)
                    # size of samples
                    n_samples = Xb.shape[0]

                    # draw a batch of random samples from the domain, same size as Xb
                    inputs_t = torch.tensor(
                        np.random.uniform(
                            low=t_low, 
                            high=t_high, 
                            size=(n_samples, 1)
                        ), 
                        requires_grad=True
                    )
                    inputs_x = torch.tensor(
                        np.random.uniform(
                            low=x_low, 
                            high=x_high, 
                            size=(n_samples, 1)
                        ), 
                        requires_grad=True
                    )
                    # uniform random samples in the interior
                    Xb2 = torch.cat(
                        [inputs_t, inputs_x], dim=-1
                    )
                    # evaluate PDE loss
                    pde_loss = self.physics_loss(Xb2, level)

                    # evaluate data loss
                    data_loss = self.data_loss(Xb, yb, level)

                    # add training loss (potentially reweight)
                    train_loss = pde_loss + data_loss

                    # save loss history for this epoch
                    all_batch_losses_data.append(data_loss.item())
                    all_batch_losses_pde.append(pde_loss.item())

                    # compute backpropagation (should change this level only)
                    train_loss.backward()

                    return train_loss
                
                # step optimizer training loss
                optim.step(closure=closure)
            # ----------
            if scheduler:
                # step scheduler after epoch if there is one
                scheduler.step()
                print("---------- \n")
                print("++ Learning rate reduced, now at = {0:0.6f}".format(scheduler.get_lr()[0]))
            
            # record epoch-wise average losses
            all_epoch_data_loss.append(np.mean(all_batch_losses_data))
            all_epoch_pde_loss.append(np.mean(all_batch_losses_pde))

        # record information for this level's training
        info = {
            "level": level,
            "pde_loss": all_epoch_pde_loss,
            "data_loss": all_epoch_data_loss
        }
        return info

    def train_aggregate(self, 
            X, y, num_cycles, 
            all_epochs, all_batch_size, 
            optims, schedulers
        ):
        """ 
            Train all neural networks, organized by level, with 
            possible layer shuffling.
        """
        assert self.num_levels > 0 
        assert len(all_epochs) == self.num_levels
        assert len(optims) == self.num_levels
        assert len(schedulers) == self.num_levels
        # report information after a number of cycles
        info = []
        for k in range(num_cycles):
            print("--------------------------------------------------\n")
            print("                  Cycle = {}                      \n")
            print("--------------------------------------------------\n")
            print("... Beginning training. \n")
            # training each level
            info_cycle = []
            for i in range(self.num_levels):
                info_i = self.train(
                    X, y, i, 
                    all_batch_size[i], 
                    all_epochs[i], 
                    optims[i], 
                    schedulers[i]
                )
                info_cycle.append(info_i)
            info.append(info_cycle)
        return info
            
            



















            