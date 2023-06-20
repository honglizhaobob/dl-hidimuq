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
        self.tgrid = self.raw_data["tgrid"].flatten()
        self.tgrid = torch.tensor(self.tgrid, requires_grad=True)

        self.tmin = self.tgrid.min()
        self.tmax = self.tgrid.max()
        self.nt = len(self.tgrid)
        self.dt = self.tgrid[1]-self.tgrid[0]

        self.xgrid = self.raw_data["xgrid"].flatten()
        self.xgrid = torch.tensor(self.xgrid, requires_grad=True)

        self.xmin = self.xgrid.min()
        self.xmax = self.xgrid.max()
        self.nx = len(self.xgrid)
        self.dx = self.xgrid[1]-self.xgrid[0]

        # flattened spatio-temporal grid
        self.inputs = cartesian_data(self.tgrid, self.xgrid)

        # flattened spatio-temporal grid without boundaries
        self.inputs_interior = cartesian_data(self.tgrid[1:], self.xgrid[1:-1])

        # (low-accuracy) solution from KDE
        self.pmc = self.raw_data["pmc"]

        # check KDE data has grid shape
        assert self.pmc.shape[0] == self.nt
        assert self.pmc.shape[1] == self.nx

        self.pmc = torch.tensor(self.pmc).T.flatten()


        

######################################################################
# Loss functions
######################################################################
    def domain_loss(self, inputs):
        """
            Physics loss on the main PDE domain, which is assumed to not include
            boundaries and initial.
        """
        # compute drift
        g_eval = self.G_nn(inputs)

        # compute diffusion
        d_eval = self.D_nn(inputs)

        # compute NN PDE solution
        p = self.p_nn(inputs)

        # time derivative and spatial derivative
        deriv = self.gradient(
            p, inputs, order=1
        )
        p_dt = deriv[:, 0][:, None]
        p_dx = deriv[:, 1][:, None]

        # d/dx ( G(t, x) * p )
        tmp = g_eval * p
        dGpdx = self.gradient(
            tmp, inputs, order=1
        )[:, 1][:, None]

        # d/dx D(t, x)
        D_dx = self.gradient(
            d_eval, inputs, order=1
        )[:, 1][:, None]

        # d2/dx2 ( D(t, x) * p ) = d/dx (d/dx D(t, x) * p + D(t, x) * p_dx)
        tmp = D_dx * p + d_eval * p_dx
        dDpdxx = self.gradient(
            tmp, inputs, order=1
        )[:, 1][:, None]

        # assemble PDE loss
        loss = torch.mean( ( p_dt + dGpdx - dDpdxx ) ** 2)
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
                2 * ( torch.log(torch.abs(p_pred - pmc)) - gamma * torch.log(torch.abs(p_pred)) )
            )
        else:
            # regular least squares
            residual = (p_pred - pmc) ** 2

        residual = torch.mean(residual)
        return residual
    
    def initial_loss(self, n_samples, p0, strategy="uniform"):
        """
            Evaluates PDE loss on the initial condition. The initial condition
            for the RO-PDF equation must be an evaluable expression.
        """
        if strategy == "uniform":
            _xgrid = torch.linspace(self.xmin, self.xmax, n_samples).reshape(-1, 1)
        elif strategy == "rand":
            _xgrid = torch.FloatTensor(n_samples).uniform_(self.xmin, self.xmax).reshape(-1, 1)
        else:
            raise NotImplementedError()
        
        # evaluate exact 
        _p0_exact = p0(_xgrid)

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
        """
            Computes NN prediction on the stored grid and ensure 
            mass conservation and non-negativity.
        """
        # make prediction on all grid points
        inputs = self.inputs
        p_nn_eval = self.p_nn(inputs)

        # reshape back to 2d solution
        p_nn_eval2d = torch.reshape(
            p_nn_eval, [self.nx, self.nt]
        ).T
        assert p_nn_eval2d.shape[0] == self.nt 
        assert p_nn_eval2d.shape[1] == self.nx

        # numerically integrate in space
        int_p_nn_eval2d_dx = torch.trapz(p_nn_eval2d, self.xgrid)

        # evaluate integral loss
        integral_loss = torch.mean(
            ( int_p_nn_eval2d_dx - torch.ones_like(int_p_nn_eval2d_dx) ) ** 2
        )

        # evaluate non-negativity constriant
        negativity_loss = torch.sum(
            (1 / (self.nt * self.nx)) * ( p_nn_eval2d[p_nn_eval2d < 0.] ) ** 2
        )

        loss = integral_loss + negativity_loss
        return loss

######################################################################
# Training routine
######################################################################
def train(
        model, optim, scheduler,
        batch_size, epochs=100, 
        early_stopping=None, 
        mode="all",
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
    X = model.inputs
    y = model.pmc.reshape(-1, 1)
    assert len(y) == X.shape[0]
    assert X.shape[1] == 2

    # number of samples
    N = X.shape[0]

    # determine number of batches
    num_batches = int(N / batch_size)
    all_epoch_losses = []
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
            X, y = X[tmp, :].data, y[tmp, :].data # the `.data` detaches computational graph

        # loop over batches
        # ----------------------------------------
        # Define batch-wise variables
        #
        all_batch_losses = []
        all_batch_losses_data = []
        all_batch_losses_pde = []
        #all_batch_losses_reg = []
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

                data_loss = None

                if mode == "all":
                    # ----------------------------------------
                    #              PDE physics
                    # ----------------------------------------

                    # generate a prediction on batch data
                    pde_loss = model.domain_loss(Xb)

                    # ----------------------------------------
                    #              Data loss
                    # ----------------------------------------
                    data_loss = model.data_loss(Xb, yb)


                elif mode == "data_only":
                    # no physics enforced
                    pde_loss = torch.tensor(0.0)
                    # ----------------------------------------
                    #              Data loss
                    # ----------------------------------------
                    data_loss = model.data_loss(Xb, yb)

                elif mode == "physics_only":
                    # ----------------------------------------
                    #             PDE physics
                    # ----------------------------------------
                    pde_loss = model.domain_loss(Xb)

                    # no data regularization enforced
                    data_loss = torch.tensor(0.0)

                else:
                    raise NotImplementedError()

                # ----------------------------------------
                #          Regularity loss
                # ----------------------------------------
                # computed separately on entire spatio-temporal grid
                #reg_loss = model.regularity_loss()        
                
                # aggregate
                train_loss = pde_loss + data_loss #+ reg_loss
                # ----------------------------------------
                #          Save history
                # ----------------------------------------
                all_batch_losses.append(train_loss.item())
                all_batch_losses_pde.append(pde_loss.item())
                all_batch_losses_data.append(data_loss.item())
                #all_batch_losses_reg.append(reg_loss.item())

                # compute backpropagation
                train_loss.backward()
                return train_loss
            
            # ----------
            if not scheduler:
                # if no scheduler, update model parameters directly
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
        if len(all_batch_losses_data) > 0:
            print("                     D Loss             = {}".format(np.mean(all_batch_losses_data)))
        #if len(all_batch_losses_data) > 0:
        #    print("                     R Loss             = {}".format(np.mean(all_batch_losses_reg)))

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