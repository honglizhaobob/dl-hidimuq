# Main driver script for an MLP approximation of PDF equations with closure. Implemented in PyTorch. 
# The reduced-order PDF equation is a deterministic advection-diffusion equation of the following form:
# 
#           p_t + d_dx ( G(t, x) * p ) = d2_dx2 ( D(t, x) * p)
# where G, D are respectively drift and diffusion coefficients that are generally depenedent on time. 
# This module assumes that p(t, x) is a 1-dimensional solution of RO-PDF.
#
# The order of input for the NN solution is assumed to be: p_nn(t_i, x_j) and data ordering
# will always be assumed as (column major order):
#               [t_0, x_0]              => p_nn(0, 0)
#               [t_1, x_0]              => p_nn(1, 0)
#                 ...
#               [t_max, x_0]            => p_nn(nt, 0)
#                 ---
#               [t_0, x_1]              => p_nn(0, 1)
#               [t_1, x_1]              => p_nn(1, 1)
#                 ...
#               [t_max, x_1]            => p_nn(nt, 1)
#                 ---
#               [t_0, x_max]            => p_nn(0, nx)
#               [t_1, x_max]            => p_nn(1, nx)
#                 ...
#               [t_max, x_max]          => p_nn(nt, nx)



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

################################################################################
# References
################################################################################
# - https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb
# - https://github.com/jlager/BINNs/blob/master/Notebooks/BINNsTraining.ipynb
######################################################################

class DNN(nn.Module):
    # can add dropout
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        return self.layers(x)

class G_Net(nn.Module):
    """ 
        A densely connected neural network surrogate 
        for the solution of PDF equation with closure
        approximation.

        By default holds a DNN with 2 hidden layers, 
        each of which has 64 nodes, and `tanh` activation function.
    """
    def __init__(self, layers=[2, 64, 64, 1]):
        super(G_Net, self).__init__()
        self.net = DNN(layers)

    def forward(self, inputs):
        return self.net(inputs)

class D_Net(nn.Module):
    """ 
        A densely connected neural network surrogate for 
        the unknown forcing term arising from the Reynolds'
        decomposition. 

        By default, it is a DNN with 1 hidden layer, each of which
        has 32 nodes, and `tanh` activation function.

    """
    def __init__(self, layers=[2, 32, 1]):
        super(D_Net, self).__init__()
        self.net = DNN(layers)

    def forward(self, inputs):
        return self.net(inputs)

class P_Net(nn.Module):
    """
        A densely connected neural network surrogate for the 
        solution for the RO-PDF equation.
    """
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super(P_Net, self).__init__()
        self.net = DNN(layers)
    def forward(self, inputs):
        return self.net(inputs)

######################################################################
# PINN
######################################################################
class PhysicsInformedROPDF(nn.Module):
    def __init__(self, indim, outdim, data_path):
        super(PhysicsInformedROPDF, self).__init__()
        self.indim = indim
        self.outdim = outdim
        # load data
        self.load_data(data_path) 
        # initialize NNs
        self.build_models()
        # initialize optimizer
        self.optimizer = torch.optim.LBFGS(
            self.parameters(), 
            lr=2.0, 
            max_iter=10000, 
            max_eval=10000, 
            history_size=30,
            tolerance_grad=1e-8, 
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe"
        ) # Adam

        # number of samples for evaluating PDE loss
        self.num_pde_samples = 10000

    def forward(self, inputs):
        """ 
            Generate approximate PDE solution. Input
            `X` should be grid data of form [x, t]. 
        """
        return self.p_nn(inputs)
    
    def gradient(self, outputs, inputs, order=1):
        """ 
            Helper function for taking gradient up to `order`
            of output with respect to input.
        """
        grads = outputs
        outputs = outputs.sum()
        # compute gradients sequentially until order is reached
        for i in range(order):
            grads = torch.autograd.grad(outputs, inputs, create_graph=True)[0]
            outputs = grads.sum()
        return grads


    def build_models(self):
        """ Initialize neural nets. """

        # neural nets for coefficients
        self.G_nn = G_Net()
        self.D_nn = D_Net()

        # solution neural net
        self.p_nn = P_Net()

    def load_data(self, path):
        """
            Loads a low-fidelity kernel density estimate of the density
            at different times.

            The data contained in `path` is assumed to have the following
            fields:

            data[`t_grid`]                  1d time grid.
            data[`x_grid`]                  1d spatial grid.
            data[`pmc`]                     (Nt x Nx) array containing 
                                            a solution estimate at (t_i, x_j).
        """
        self.raw_data = scipy.io.loadmat(path)
        # unpack variables

        # PDE domain
        self.tgrid = self.raw_data["t_grid"].reshape(-1, 1)
        self.tmin = self.tgrid.min()
        self.tmax = self.tgrid.max()
        self.nt = len(self.tgrid)
        self.dt = self.tgrid[1]-self.tgrid[0]

        self.xgrid = self.raw_data["x_grid"].reshape(-1, 1)
        self.xmin = self.xgrid.min()
        self.xmax = self.xgrid.max()
        self.nx = len(self.xgrid)
        self.dx = self.xgrid[1]-self.xgrid[0]

        # (low-accuracy) solution from KDE
        self.pmc = self.raw_data["pmc"]
        assert self.pmc.shape[0] == self.nt
        assert self.pmc.shape[1] == self.nx

######################################################################
# Loss functions
######################################################################
    def domain_loss(self, inputs, outputs, advection_diffusion_data=None, fresh_samples=False):
        pass

    def data_loss(self, inputs, pmc, gamma=0.0, ic_weight=1/4, lbc_weight=1/4, rbc_weight=1/4):
        """
            Given `inputs`, and output `pmc`, evaluates neural net
            solution at `inputs` and compute (generalized) least square
            loss on the predictions.

            The KDE data is assumed to contain initial condition and 
            boundary condition. During evaluation, they can be potentially 
            weighted differently.

            Note that if `self.data_loss()` is used, there will be no need to
            include `self.boundary_loss()` and `self.initial_loss()`. It is
            assumed that we are in the data-rich regime whenever `self.data_loss()`
            is used. 


            Inputs:
                inputs                          Flattened temporal-spatial 
                                                locations where we have an 
                                                estimate of the solution.
                
                pmc                             Flattened estimate values 
                                                corresponding locations in `inputs`.
                
                gamma                           Generalized least squares exponent,
                                                defaults to 0.0 (standard LS)
                
                ic_weight, bc_weight            Weights of initial condition and 
                                                boundary condition for final loss sum.
                                                `interior_weight` = 
                                                    1.0 - ic_weight - (lbc_weight + rbc_weight)

                                                Defaults to equal weights.  
        """
        n_samples = len(pmc)
        assert inputs.shape[0] == n_samples
        assert inputs.shape[1] == 2 # time and 1d space

        # find indices of left boundary, right boundary, initial condition
        initial_data_locs = (inputs[:, 0][:, None] == self.tmin)
        left_boundary_data_locs = (inputs[:, 1][:, None] == self.xmin)
        right_boundary_data_locs = (inputs[:, 1][:, None] == self.xmax)
        internal_data_locs = torch.logical_not(
            torch.logical_or(
                initial_data_locs, 
                torch.logical_or(
                    left_boundary_data_locs, 
                    right_boundary_data_locs
                )
            )
        )

        # compute residual 
        p = self.p_nn(inputs)
        if gamma:
            # generalized least squares (stable division)
            residual = torch.exp(
                2 * ( torch.log((p - pmc)) - gamma * torch.log(torch.abs(p)) )
            )
        else:
            # regular least squares
            residual = (p - pmc) ** 2

        # modify weights of different contributions

        # initial condition
        residual *= torch.where(
            initial_data_locs,
            ic_weight * torch.ones_like(residual),
            torch.ones_like(residual)
        )

        # left boundary 
        residual *= torch.where(
            left_boundary_data_locs,
            lbc_weight * torch.ones_like(residual),
            torch.ones_like(residual)
        )

        # right boundary
        residual *= torch.where(
            right_boundary_data_locs,
            rbc_weight * torch.ones_like(residual),
            torch.ones_like(residual)
        )

        # internal
        residual *= torch.where(
            internal_data_locs,
            (1 - ic_weight - (lbc_weight + rbc_weight)) * torch.ones_like(residual),
            torch.ones_like(residual)
        )
        residual = torch.mean(residual)
        return residual
    
    def boundary_loss(self, n_samples, strategy="uniform"):
        """
            Evaluates PDE loss on the boundary. The boundary for the 
            RO-PDF equation is assumed to be vanishing.

            Inputs:
                n_samples                   number of spatial locations to query. 
                                            The effective data size will be 
                                            `n_samples * self.nt`
                strategy                    strategy used to generate points on the 
                                            boundary. 
                                            
                                            Defaults to `uniform` where 
                                            the time grid is discretized with a uniform
                                            mesh. Other strategies include:

                                                `rand`      sample points uniformly randomly
                                                            from interval [tmin, tmax]
        """
        if strategy == "uniform":
            _tgrid = torch.linspace(self.tmin, self.tmax, n_samples).reshape(-1, 1)
        elif strategy == "rand":
            _tgrid = torch.FloatTensor(n_samples).uniform_(self.tmin, self.tmax).reshape(-1, 1)
        else: 
            raise NotImplementedError()
        # left boundary: repeat `xmin`
        _xmin = torch.tensor([self.xmin]).repeat(n_samples).reshape(-1, 1)

        # right boundary: repeat `xmax`
        _xmax = torch.tensor([self.xmax]).repeat(n_samples).reshape(-1, 1)

        # concatenate to form grid data
        _left_inputs = torch.concat(
            [_tgrid, _xmin], dim=1
        )
        _right_inputs = torch.concat(
            [_tgrid, _xmax], dim=1
        )

        # evaluate (by default standard least squares loss)
        _left_loss = torch.mean(
            self.p_nn(_left_inputs) ** 2
        )
        _right_loss = torch.mean(
            self.p_nn(_right_inputs) ** 2
        )
        loss = _left_loss + _right_loss
        return loss
    
    def regularity_loss(self):
        pass
    
    def loss(self, y_pred, y_true, randomized=False, weighting=False):
        """ 
            Compute aggregate loss function from a 
            suite of loss functions.

            If loss `weighting` is not `False`, need to input a 
            vector of weights (not necessarily summing to 1.0).

            The ordering of losses should be [pde, data, monotonicity]
        """
        pass
    

######################################################################
# Training routine
######################################################################
def train(
        X, y, 
        model, optim, scheduler,
        batch_size, epochs, 
        early_stopping=None, 
        mode="all",
        shuffle=True,
        batch_print=50
    ):
    """ 
        Trains a PINN-CDF model using an optimizer.


        `X` should contain all queried locations where we
        evaluate the approximation to the solution of the CDF
        equation, `Fnn`, organized as a (N_pnts x 2) torch.tensor

        Similarly, `y` should be of size (N_pnts x 1) torch.tensor

        Inputs:

        Outputs:

    """
    # number of samples
    N = X.shape[0]
    # determine number of batches
    num_batches = int(N/batch_size)
    all_epoch_losses = []
    for i in range(epochs):
        print("-------------- Epoch {} ------------ \n".format(i+1))
        # turn on training mode
        model.train()
        # randomly shuffle training data before each epoch
        if shuffle:
            # generate permutation indices
            tmp = np.random.permutation(N)
            X, y = X[tmp, :].data, y[tmp, :].data # the `.data` detaches computational graph
            advection = model.pde_data[1][tmp, :][:, None].data
            diffusion = model.pde_data[2][tmp, :][:, None].data
        # loop over batches
        # ----------------------------------------
        # Define batch-wise variables
        #
        all_batch_losses = []
        all_batch_losses_data = []
        all_batch_losses_pde = []
        all_batch_losses_monotone = []
        # ----------------------------------------

        for idx in range(num_batches):
            # ----------

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
                advection_batch = advection[start_idx:end_idx, :].data.clone()
                diffusion_batch = diffusion[start_idx:end_idx, :].data.clone()
                advection_diffusion_data = torch.cat([advection_batch, diffusion_batch], dim=1)
                
                # require gradients
                Xb.requires_grad = True

                # generate a prediction on batch data
                yb_pred = model.forward(Xb)

                data_loss = None
                if mode == "data_only" or mode == "all":
                    # ----------------------------------------
                    # Data loss
                    # ----------------------------------------
                    data_loss, initial_loss, \
                        left_boundary_loss, right_boundar_loss = model.data_loss(Xb, yb_pred, yb)
                    all_batch_losses_data.append(data_loss.item())
                    # print and report more granular data loss terms
                    print_flag = False
                    if left_boundary_loss.item() != 0 or right_boundar_loss.item() != 0 or initial_loss.item() != 0:
                        print_flag = True
                    if print_flag and idx % batch_print == 0:
                        print("\n <<< Batch = {}, Data Loss Profile >>>".format(idx+1))
                        print("\n ==============================\n")
                        print(" -- initial condition = {}".format(initial_loss))
                        print(" -- left boundary condition = {}".format(left_boundary_loss))
                        print(" -- right boundary condition = {}".format(right_boundar_loss))
            
                pde_loss, m_loss = None, None
                if mode == "physics_only" or mode == "all":
                    # ----------------------------------------
                    # PDE loss + monotonicity loss
                    # ----------------------------------------

                    # note that this part of the loss function is
                    # independent of batch data
                    pde_loss, m_loss = model.domain_loss(Xb, yb_pred, advection_diffusion_data=advection_diffusion_data)
                    all_batch_losses_pde.append(pde_loss.item())
                    all_batch_losses_monotone.append(m_loss.item())

                if idx % batch_print == 0:
                    print("      Batch {}, Num Batches = {}          ".format(idx + 1, num_batches))
                # add up all the loss and printing (!!! perhaps weighted)
                if mode == "data_only":
                    train_loss = data_loss
                elif mode == "physics_only":
                    train_loss = pde_loss + m_loss
                else:
                    train_loss = 10*data_loss + pde_loss + m_loss
                
                # save loss history
                all_batch_losses.append(train_loss.item())
                # compute backpropagation
                train_loss.backward()
                #print(">> PDE loss = {}, M loss = {}, D loss = {}".format(pde_loss.item(), m_loss.item(), data_loss.item()))
                return train_loss

            # ----------

            if scheduler is None:
                # update model parameters
                optim.step(closure=closure)
            else:
                # step scheduler
                scheduler.step(closure())
        
        # print epoch statistics
        epoch_loss = np.mean(all_batch_losses)
        # early stopping 
        if early_stopping is not None:
            if epoch_loss <= early_stopping:
                print("Early stopping ... | Loss(u) | <= {} \n\n".format(early_stopping))
                break
        print("------------------------------------------------------------------\n")
        print("|        Epoch {}, Batch Average Loss = {}                       |\n".format(i+1, epoch_loss))
        print("------------------------------------------------------------------\n")
        if len(all_batch_losses_pde) > 0:
            print("                     P Loss             = {}".format(np.mean(all_batch_losses_pde)))
        if len(all_batch_losses_monotone) > 0:
            print("                     M Loss             = {}".format(np.mean(all_batch_losses_monotone)))
        if len(all_batch_losses_data) > 0:
            print("                     D Loss             = {}".format(np.mean(all_batch_losses_data)))
            # post processing

######################################################################
# Postprocessing, prediction, and model validation. 
######################################################################
if __name__ == "__main__":
    # debug
    pinn = PhysicsInformedROCDF(2, 1, Fmc_data_path="/Fqoi.mat")
    X = pinn.pde_data[0]
    y = pinn.fmc_data

    