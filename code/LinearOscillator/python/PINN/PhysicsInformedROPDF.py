# Main driver script for an MLP approximation of PDF equations with closure. Implemented in PyTorch. 
# The reduced-order PDF equation is a deterministic advection-diffusion equation of the following form:
# 
#           p_t + d_dx ( G(t, x) * p ) = d2_dx2 ( D(t, x) * p)
# where G, D are respectively drift and diffusion coefficients that are generally depenedent on time. 
# This module assumes that p(t, x) is a 1-dimensional solution of RO-PDF.
#
#
# This routine can also learn the CDF equation. In the case of CDF equation, the advection
# will be in non-conservative form.
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

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

################################################################################
# References
################################################################################
# - https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb
# - https://github.com/jlager/BINNs/blob/master/Notebooks/BINNsTraining.ipynb
######################################################################

class DNN(nn.Module):
    def __init__(
        self, layers, 
        activation=torch.nn.ReLU, 
        last_layer_activation=torch.nn.ReLU
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
        
    def forward(self, x):
        return self.layers(x)

class G_Net(nn.Module):
    """ 
        A densely connected neural network surrogate 
        for the solution of PDF equation with closure
        approximation.

        By default holds a DNN with 3 hidden layers, 
        each of which has 64 nodes, and `tanh` activation function.
    """
    def __init__(self, layers=[2, 32, 32, 32, 1], activation=torch.nn.Tanh):
        super(G_Net, self).__init__()

        # no activation used in output layer
        self.net = DNN(
            layers, 
            activation=activation, 
            last_layer_activation=None
            #last_layer_activation=torch.nn.Softplus
        )

        # testing constant
        self.constant = torch.nn.Parameter(torch.tensor([0.05]))

        # regression data (target)
        self.regression_data = None

    def forward(self, inputs):
        return self.net(
           inputs
        )
    
    def load_data(self, path):
        if not self.regression_data and path is not None:
            # load target data
            raw_data = scipy.io.loadmat(path)
            targ = raw_data["cond_exp"]

            # load input data

        else:
            raise NotImplementedError()
    
    def loss(self, inputs, y_exact):
        """
            Regression function on data `y_exact`, makes 
            prediction using `inputs` and computes a 
            standard least-squares loss. 
        """
        y_pred = self.net(inputs)
        loss = torch.mean( (y_pred - y_exact) ** 2 )
        return loss

class D_Net(nn.Module):
    """ 
        A densely connected neural network surrogate for 
        the unknown forcing term arising from the Reynolds'
        decomposition. 

        By default, it is a DNN with 3 hidden layers, each of which
        has 32 nodes, and `tanh` activation function.

    """
    def __init__(self, layers=[2, 32, 32, 32, 1], activation=torch.nn.ReLU):
        super(D_Net, self).__init__()

        # output layer is constrained to be non-negative
        self.net = DNN(
            layers, 
            activation=activation,
            last_layer_activation=torch.nn.Softplus
        )

    def forward(self, inputs):
        return self.net(inputs)

    def loss(self, inputs, y_exact):
        """
            Regression function on data `y_exact`, makes 
            prediction using `inputs` and computes a 
            standard least-squares loss. 
        """
        y_pred = self.net(inputs)
        loss = torch.mean( (y_pred - y_exact) ** 2 )
        return loss

class P_Net(nn.Module):
    """
        A densely connected neural network surrogate for the 
        solution for the RO-PDF equation.

    """
    def __init__(self, layers=[2, 128, 128, 128, 1], activation=torch.nn.ReLU):
        super(P_Net, self).__init__()

        self.net = DNN(
            layers, activation=activation, 
            last_layer_activation=None
            # None or may choose torch.nn.Softplus to constrain >= 0 solutions.
        )
    def forward(self, inputs):
        return self.net(inputs)

######################################################################
# PINN
######################################################################
class PhysicsInformedROPDF(nn.Module):
    def __init__(
        self, indim, outdim, data_path,
        scheduler=None, optimizer="adam"
    ):
        super(PhysicsInformedROPDF, self).__init__()
        self.indim = indim
        self.outdim = outdim
        
        # load data
        self.normalize = False
        self.load_data(data_path) 
        # initialize NNs
        self.build_models()
        # initialize optimizer

        if optimizer == "adam":
            # Adam: 
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=8e-3
            )
        elif optimizer == "lbfgs":
            # L-BFGS
            self.optimizer = torch.optim.LBFGS(
            self.parameters(), 
            lr=1e-3, 
            max_iter=10000, 
            max_eval=10000, 
            history_size=30,
            tolerance_grad=1e-8, 
            tolerance_change=1e-10,
            line_search_fn="strong_wolfe"
        ) # L-BFGS
        else:
            raise NotImplementedError()

        # scheduler
        if scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)
        elif scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.1)
        else:
            self.scheduler = None

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
        """ 
            Initialize neural nets. 
        """
        # neural nets for coefficients
        self.G_nn = G_Net()
        self.D_nn = D_Net()

        # solution neural net
        self.p_nn = P_Net()


    def load_data(self, path, normalize=True, normalize_output=False):
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
        self.tgrid = self.raw_data["tgrid"].flatten()
        self.xgrid = self.raw_data["xgrid"].flatten()

        self.tmin = self.tgrid.min()
        self.tmax = self.tgrid.max()

        self.xmin = self.xgrid.min()
        self.xmax = self.xgrid.max()

        # normalize to unit scale
        if normalize:
            self.normalize = True
            self.xgrid = self.xgrid / self.xmax
            self.tgrid = self.tgrid / self.tmax

        self.tgrid = torch.tensor(self.tgrid, requires_grad=True)
        self.xgrid = torch.tensor(self.xgrid, requires_grad=True)
        self.nx = len(self.xgrid)
        self.dx = self.xgrid[1]-self.xgrid[0]
        self.nt = len(self.tgrid)
        self.dt = self.tgrid[1]-self.tgrid[0]

        # flattened spatio-temporal grid without boundaries
        self.inputs_interior = cartesian_data(self.tgrid[1:], self.xgrid[1:-1])

        # (low-accuracy) solution from KDE only on interior points
        self.pmc = self.raw_data["pmc"][1:, 1:-1]
        # check KDE data has grid shape
        assert self.pmc.shape[0] == self.nt - 1 # without initial condition
        assert self.pmc.shape[1] == self.nx - 2 # without 1d boundaries

        # flatten in column major order to form training data
        self.pmc = torch.tensor(self.pmc).T.flatten()
        if normalize_output:
            self.pmc_mean = torch.mean(self.pmc)
            self.pmc = self.pmc / self.pmc_mean

        # initial condition, specified as an array
        self.p0 = self.raw_data["pmc"][0, :].flatten()
        self.p0 = torch.tensor(self.p0, requires_grad=False)

        # reporting
        print("------------------------------------------------------------\n")
        print("=> Data Loaded at: {} \n".format(path))
        if normalize:
            if not normalize_output:
                self.pmc_mean = None
            print("Data is normalized, on scale (t, x)=>p -- ({}, {})=>{}".format(self.tmax, self.xmax, self.pmc_mean))
        print("----> Total Number of observations = {} \n".format(len(self.pmc)))
        print("------------------------------------------------------------\n")



######################################################################
# Loss functions
######################################################################
    def domain_loss(self, n_samples=2**12, mode="advection", conservative=False):
        """
            Physics loss on the main PDE domain. The input points are 
            generated by random uniform sampling.

            Two modes are supported:
                1. "advection": only advection term is assumed in the PDE
                2. "advection_diffusion": includes both advection and diffusion

            `conservative` specifies whether the PDE model is in conservative form.

        """
        if self.normalize:
            t_low = self.tmin / self.tmax
            t_high = 1.0
            x_low = self.xmin / self.xmax
            x_high = 1.0
        else:
            t_low = self.tmin
            t_high = self.tmax
            x_low = self.xmin
            x_high = self.xmax

        # uniformly sample inputs
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
        inputs = torch.cat(
            [inputs_t, inputs_x], dim=-1
        )

        # evaulate NN predictions
        p = self.p_nn(inputs)

        # time derivative and spatial derivative
        deriv = self.gradient(
            p, inputs, order=1
        )
        p_dt = deriv[:, 0][:, None]
        p_dx = deriv[:, 1][:, None]

        # predicting advection coefficients
        g_eval = self.G_nn(inputs)

        if conservative:
            # d/dx ( G(t, x) * p ) - conservative form
            tmp = g_eval * p
            dGpdx = self.gradient(
                tmp, inputs, order=1
            )[:, 1][:, None]
        else:
            # G(t, x) * dp/dx - non-conservative form
            dGpdx = g_eval * p_dx

        # add advection contribution

        # need to rescale advection if we scaled the inputs, only correct in conservative form
        if self.normalize and conservative:
            peclet = (self.tmax / self.xmax)
        else:
            peclet = 1.0
        residual = p_dt + peclet * dGpdx

        # predicting diffusion coefficients if enabled
        if mode == "advection_diffusion":
            # compute diffusion
            d_eval = self.D_nn(inputs)
            # d/dx D(t, x)
            D_dx = self.gradient(
                d_eval, inputs, order=1
            )[:, 1][:, None]

            # d2/dx2 ( D(t, x) * p ) = d/dx (d/dx D(t, x) * p + D(t, x) * p_dx)
            tmp = D_dx * p + d_eval * p_dx
            dDpdxx = self.gradient(
                tmp, inputs, order=1
            )[:, 1][:, None]
            residual = residual - dDpdxx

        # assemble PDE loss
        loss = torch.mean( residual ** 2 )
        return loss
        

    def data_loss(self, inputs, pmc, gamma=0.0):
        """
            Given `p_pred` and `pmc` at corresponding locations, 
            compute the (generalized) least-squares loss with parameter
            `gamma`.  
        """
        n_samples = len(pmc)
        assert n_samples == inputs.shape[0]
        assert inputs.shape[1] == 2

        # make prediction separately
        p_pred = self.p_nn(inputs)

        if gamma:
            # generalized least squares (stable division)
            residual = torch.exp(
                ( torch.log(torch.abs(p_pred - pmc)) - gamma * torch.log(torch.abs(p_pred)) )
            )
        else:
            # mean-squared loss
            residual = (p_pred - pmc)

        residual = torch.mean(residual ** 2)
        return residual
    
    def initial_loss(self, n_samples=500):
        """
            Evaluates loss on the initial time t = 0 at randomly queried locations
            on the uniform grid `self.xgrid`. The initial condition is specified 
            as discrete values. 
        """
        # sample random values from the spatial grid (possibly with replacements)
        rand_inds = np.random.choice(self.nx, n_samples)
        _xgrid = self.xgrid.clone().detach()[rand_inds].reshape(-1, 1)

        # query exact values
        _p0_exact = self.p0.clone().detach()[rand_inds]

        # append time 
        _t0 = torch.tensor([self.tmin]).repeat(n_samples).reshape(-1, 1)
        _inputs = torch.concat(
            [_t0, _xgrid], dim=1
        )

        # evaluate NN predictions 
        _p0_nn = self.p_nn(_inputs)

        # compute loss
        loss = torch.mean(
            (_p0_nn - _p0_exact) ** 2
        )
        return loss

    def boundary_loss(self, n_samples=500, strategy="uniform"):
        """
            Evaluates PDE loss on the boundary. The boundary for the 
            RO-PDF equation is assumed to be vanishing.

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
    
    # helper methods
    def _boundary_data(self):
        """
            Helper subroutine to get boundary data in input format, size is 
            assumed to be (Nt x 2) (left and right boundary for all time points).
        """
        x_grid = torch.tensor([
            self.xmin, self.xmax
        ], requires_grad=True)
        t_grid = self.tgrid.clone()
        boundary_data = cartesian_data(t_grid, x_grid)
        return boundary_data

    
######################################################################
# Training routine
######################################################################
def train(
        model, optim, scheduler,
        batch_size, epochs=50, 
        early_stopping=None, 
        mode="data_only",
        shuffle=True,
        batch_print=50
    ):
    """ 
        Trains a PINN-PDF model using an optimizer.

        `X` should contain all queried locations where we
        evaluate the approximation to the solution of the CDF
        equation, `Fnn`, organized as a (N_pnts x 2) torch.tensor

        Similarly, `y` should be of size (N_pnts x 1) torch.tensor

    """
    # PDE loss is only evaluated at interior points
    X = model.inputs_interior
    y = model.pmc.reshape(-1, 1)
    assert len(y) == X.shape[0]
    assert X.shape[1] == 2

    # number of samples
    N = X.shape[0]

    # determine number of batches
    num_batches = int(N / batch_size)

    # epoch-wise losses (averaged over batch)
    all_epoch_pde_loss = []
    all_epoch_data_loss = []
    all_epoch_init_loss = []
    all_epoch_boundary_loss = []
    for i in range(epochs):
        print("------------------------------------------------------------------\n")
        print("|                      Epoch {}                                  |\n".format(i+1))
        print("------------------------------------------------------------------\n")
        # turn on training mode
        model.train()
        # randomly shuffle training data before each epoch
        if shuffle:
            # generate permutation indices
            tmp = np.random.permutation(N)
            # the `.data` detaches computational graph
            X, y = X[tmp, :].data.clone(), y[tmp, :].data.clone()

        # loop over batches
        # ----------------------------------------
        # Define batch-wise variables
        #
        all_batch_losses = []
        all_batch_losses_data = []
        all_batch_losses_pde = []
        all_batch_losses_reg = []
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
                
                # require gradients
                Xb.requires_grad = True

                if mode == "all":
                    # ----------------------------------------
                    #              PDE physics
                    # ----------------------------------------

                    # generate a prediction on batch data
                    pde_loss = model.domain_loss()

                    # ----------------------------------------
                    #              Data loss
                    # ----------------------------------------
                    data_loss = model.data_loss(Xb, yb)

                    # save loss history
                    all_batch_losses_pde.append(pde_loss.item())
                    all_batch_losses_data.append(data_loss.item())

                    # backpropagation
                    train_loss = pde_loss + data_loss

                elif mode == "data_only":
                    # no physics enforced
                    
                    # ----------------------------------------
                    #              Data loss
                    # ----------------------------------------
                    data_loss = model.data_loss(Xb, yb)

                    # save loss history
                    all_batch_losses_data.append(data_loss.item())

                    # backpropagation
                    train_loss = data_loss

                elif mode == "physics_only":
                    # ----------------------------------------
                    #             PDE physics
                    # ----------------------------------------
                    pde_loss = model.domain_loss()

                    # no data regularization enforced

                    # save loss history
                    all_batch_losses_pde.append(pde_loss.item())

                    # backpropagation
                    train_loss = pde_loss

                else:
                    raise NotImplementedError()
                

                # compute backpropagation
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

        # # print epoch statistics
        # epoch_loss = np.mean(all_batch_losses)
        # # early stopping 
        # if early_stopping is not None:
        #     if epoch_loss <= early_stopping:
        #         print("Early stopping ... | Loss(u) | <= {} \n\n".format(early_stopping))
        #         break
        # print("------------------------------------------------------------------\n")
        # print("|        Epoch {}, Batch Average Loss = {}                       |\n".format(i+1, epoch_loss))
        # print("------------------------------------------------------------------\n")
        # if len(all_batch_losses_pde) > 0:
        #     mean_batch_pde_loss = np.mean(all_batch_losses_pde)
        #     print("                     P Loss             = {}".format(mean_batch_pde_loss))# + mean_batch_init_loss + mean_batch_boundary_loss))
        #     print("                     |    domain        = {}".format(mean_batch_pde_loss))
        #     #print("                     |    init          = {}".format(mean_batch_init_loss))
        #     #print("                     |    bound         = {}".format(mean_batch_boundary_loss))
        #     # save 
        #     #all_epoch_init_loss.append(mean_batch_init_loss)
        #     #all_epoch_boundary_loss.append(mean_batch_boundary_loss)
        #     all_epoch_pde_loss.append(mean_batch_pde_loss)
        # if len(all_batch_losses_data) > 0:
        #     mean_batch_data_loss = np.mean(all_batch_losses_data)
        #     print("                     D Loss             = {}".format(mean_batch_data_loss))
        #     # save
        #     all_epoch_data_loss.append(mean_batch_data_loss)

        # if len(all_batch_losses_reg) > 0:
        #     mean_batch_reg_loss = np.mean(all_batch_losses_reg)
        #     print("                     R Loss             = {}".format(mean_batch_reg_loss))
        #     # save
        #     all_epoch_reg_loss.append(mean_batch_reg_loss)
        
        if all_batch_losses_pde:
            mean_batch_pde_loss = np.mean(all_batch_losses_pde)
            print("Batch PDE Loss = {} \n".format(mean_batch_pde_loss))
            # save to epoch loss
            all_epoch_pde_loss.append(mean_batch_pde_loss)
        if all_batch_losses_data:
            mean_batch_data_loss = np.mean(all_batch_losses_data)
            print("Batch Data Loss = {} \n".format(mean_batch_data_loss))
            # save to epoch loss
            all_epoch_data_loss.append(mean_batch_data_loss)

    # save info and return 
    info = {
        "pde_loss": all_epoch_pde_loss, 
        "data_loss": all_epoch_data_loss
    }
    return info


######################################################################
# Postprocessing, prediction, and model validation. 
######################################################################





######################################################################
# Helper methods
######################################################################
def cartesian_data(x, y):
    """
        Given two 1d arrays of points, return their Carteisan product
        as a list, assuming column-major flattening.
    """
    nx, ny = len(x), len(y)
    y_mesh, x_mesh = torch.meshgrid(y, x, indexing=None)
    x_mesh_flat = x_mesh.ravel().reshape(-1, 1)
    y_mesh_flat = y_mesh.ravel().reshape(-1, 1)
    res = torch.cat(
        [x_mesh_flat, y_mesh_flat], dim=1
    )
    assert len(res) == nx * ny
    return res