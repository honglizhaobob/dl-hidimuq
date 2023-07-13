# Main driver script for an MLP approximation of advection equation, where the form
# of the PDE is assumed to be:
#
#           f_dt + G(x) * f_dx = 0
# with zero Dirichlet boundary.
# where G(x) is approximated using a spatially dependent neural network. 
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

# custom libraries
from PINN.utils.helpers import cartesian_data

################################################################################
# References
################################################################################
# - https://github.com/jayroxis/PINNs/blob/master/Burgers%20Equation/Burgers%20Inference%20(PyTorch).ipynb
# - https://github.com/jlager/BINNs/blob/master/Notebooks/BINNsTraining.ipynb
######################################################################

##########################################################################################
# Data simulation
##########################################################################################
class LinearAdvection1d:
    """
        Implements the upwind scheme: https://peymandavvalo.github.io/linear_1d_advection_equation.html

    """
    def __init__(self, v, tgrid, xgrid):
        self.tgrid = tgrid
        self.xgrid = xgrid
        self.dt = tgrid[1]-tgrid[0]
        self.dx = xgrid[1]-xgrid[0]
        self.nt = len(tgrid)
        self.nx = len(xgrid)
        # coefficients on the spatial grid
        self.v = v(xgrid)
        # separate into upwind and downwind parts
        self.v_upwind = np.clip(self.v, a_min=0.0, a_max=None)
        self.v_downwind = np.clip(self.v, a_min=None, a_max=0.0)

        # compute and save finite difference matrix
        self.matrix_assemble()
    
    def matrix_assemble(self):
        """ Zero Dirichlet boundary is assumed. """
        factor = self.dt/self.dx
        alpha_min = self.v_downwind*factor
        alpha_max = self.v_upwind*factor
        upper_diag = -alpha_min[:-1]
        lower_diag = alpha_max[1:]
        diag = 1+alpha_min-alpha_max
        self.A = np.diag(lower_diag,-1)+np.diag(diag,0)+np.diag(upper_diag,1)
    
    def step(self, u_prev):
        return np.matmul(self.A, u_prev)

class LaxWendroffConservativeAdvection1d:
    """ 
        Implements the 1-dimensional Lax-Wendroff scheme for solving an advection
        PDE in conservative form, using TVD limiter. Input spatial grid is defined at 
        cell centers. In 1d, we use 2 ghost cells at each boundary.

        Zero Dirichlet boundary is assumed.

        Currently has trouble running.
    """
    def __init__(self, v, tgrid, xgrid, num_ghost=2):
        self.ng = num_ghost
        self.v = v
        self.tgrid = tgrid
        self.xgrid = xgrid
        self.xmin = min(xgrid)
        self.xmax = max(xgrid)
        self.tmax = max(tgrid)
        self.nt = len(self.tgrid)
        self.nx = len(self.xgrid)
        self.dt = self.tgrid[1]-self.tgrid[0]
        self.dx = self.xgrid[1]-self.xgrid[0]
        # spatial grid with ghost cells
        self.xgrid_effective = np.concatenate(
            [[self.xmin-self.dx*2, self.xmin-self.dx], # left boundary + 2 ghost cells
            self.xgrid, 
            [self.xmax+self.dx, self.xmax+2*self.dx]]  # right boundary + 2 ghost cells
        )
        assert len(self.xgrid_effective) == self.nx+4
        # indices of non-ghost cells
        self.xgrid_inds = np.arange(2, self.nx+2)
        # left cell edges for spatial grid
        self.xgrid_left_edges = self.xgrid_effective[2:-1]-self.dx/2 
        assert len(self.xgrid_left_edges) == self.nx + 1
        # velocity coefficient is defined at left cell edges
        self.v_left_edges = self.v(self.xgrid_left_edges)
        # positive and negative advection speeds
        self.v_p = np.clip(self.v_left_edges, a_min=0.0, a_max=None)
        self.v_m = np.clip(self.v_left_edges, a_min=None, a_max=0.0)
        # indices of positive and negative advection speeds
        self.v_p_inds = np.where(self.v_left_edges>=0.0)[0]
        self.v_m_inds = np.where(self.v_left_edges<0.0)[0]
    
    def step(self, u_prev):
        """
            Take a single time step for the numerical solution using Lax-Wendroff scheme
            and TVD limiter.
        """
        # pad input with ghost cells
        u_prev = self.pad(u_prev)
        flux_i = 0.0
        # flux at left cell edge - F[i-1/2]
        flux_m = self.v_p[0:self.nx]*u_prev[self.xgrid_inds-1]+self.v_m[0:self.nx]*u_prev[self.xgrid_inds]
        # flux at right cell edge - F[i+1/2]
        flux_p = self.v_p[1:self.nx+1]*u_prev[self.xgrid_inds]+self.v_m[1:self.nx+1]*u_prev[self.xgrid_inds+1]
        # higher order adjustments at left cell edge [i-1/2]
        Apdq = flux_i-flux_m
        # higher order adjustments at right cell edge [i+1/2]
        Amdq = flux_p-flux_i

        # wave speeds at left and right cell edges
        Wp = u_prev[self.xgrid_inds+1]-u_prev[self.xgrid_inds]
        Wm = u_prev[self.xgrid_inds]-u_prev[self.xgrid_inds-1]

        # TVD limiter: see FVM, Randall LeVeque sect. 9.13
        theta_m = np.zeros(self.nx)
        theta_p = np.zeros(self.nx)

        # left cell edge
        xsm = self.v_m_inds[self.v_m_inds<self.nx]
        xsp = self.v_p_inds[self.v_p_inds<self.nx]
        theta_m[xsm] = (
            u_prev[self.xgrid_inds[xsm]+1]-u_prev[self.xgrid_inds[xsm]]
        )/Wm[xsm]
        theta_m[xsp] = (
            u_prev[self.xgrid_inds[xsp]-1]-u_prev[self.xgrid_inds[xsp]-2]
        )/Wm[xsp]

        # right cell edge
        xsm = self.v_m_inds[self.v_m_inds>0]-1
        xsp = self.v_p_inds[self.v_p_inds>0]-1
        theta_p[xsm] = (
            u_prev[self.xgrid_inds[xsm]+2]-u_prev[self.xgrid_inds[xsm]+1]
        )/Wp[xsm]
        theta_m[xsp] = (
            u_prev[self.xgrid_inds[xsp]]-u_prev[self.xgrid_inds[xsp]-1]
        )/Wp[xsp]

        # compute modified wave
        phip = np.clip(
            (1+theta_p)/2, a_min=None, a_max=2.0
        )
        phip = np.minimum(
            phip, 2*theta_p
        )
        phip = np.clip(
            phip, a_min=0.0, a_max=None
        )

        phim = np.clip(
            (1+theta_m)/2, a_min=None, a_max=2.0
        )
        phim = np.minimum(
            phim, 2*theta_m
        )
        phim = np.clip(
            phim, a_min=0.0, a_max=None
        )

        mWp = phip*Wp
        mWm = phim*Wm
        # second order limited corrections
        Fp = 0.5*np.abs(self.v_left_edges[1:self.nx+1])*(1. - (self.dt/self.dx)*np.abs(self.v_left_edges[1:self.nx+1]))*mWp
        Fm = 0.5*np.abs(self.v_left_edges[0:self.nx])*(1. - (self.dt/self.dx)*np.abs(self.v_left_edges[0:self.nx]))*mWm
        
        # solution without ghost cells
        u_new = u_prev[self.xgrid_inds]-(self.dt/self.dx)*(Apdq+Amdq+Fp-Fm)
    
    def pad(self, u):
        """ Given solution on the spatial grid (cell centers), pad with ghost 
        cells, zero Dirichlet boundary condition is assumed. """
        assert len(u) == self.nx
        u = np.concatenate([
            [0.]*self.ng, 
            u,
            [0.]*self.ng
        ])
        return u




################################################################################


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
    def __init__(self, layers=[1, 32, 32, 32, 1], activation=torch.nn.Tanh):
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
class AdvectionNet(nn.Module):
    def __init__(
        self, indim, outdim, data_path,
        scheduler=None, optimizer="adam"
    ):
        super(AdvectionNet, self).__init__()
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
                lr=1e-3
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

        # solution neural net
        self.p_nn = P_Net()


    def load_data(self, path, normalize=False, normalize_output=False):
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

        # predicting advection coefficients (as a function of `x`)
        g_eval = self.G_nn(inputs_x)

        if conservative:
            # d/dx ( G(x) * p ) - conservative form
            tmp = g_eval * p
            dGpdx = self.gradient(
                tmp, inputs, order=1
            )[:, 1][:, None]
        else:
            # G(t, x) * dp/dx - non-conservative form
            dGpdx = g_eval * p_dx

        # need to rescale advection if we scaled the inputs, only correct in conservative form
        if self.normalize and conservative:
            peclet = (self.tmax / self.xmax)
        else:
            peclet = 1.0
        residual = p_dt + peclet * dGpdx

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
    
######################################################################
# Training routine
######################################################################
def train(
        model, optim, scheduler,
        batch_size, epochs=50, 
        early_stopping=None, 
        mode="data_only",
        shuffle=True,
        batch_print=50,
        conservative_pde=False
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
                    pde_loss = model.domain_loss(conservative=conservative_pde)

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
                    pde_loss = model.domain_loss(conservative=conservative_pde)

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

