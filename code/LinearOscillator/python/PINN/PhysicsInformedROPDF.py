# Main driver script for an MLP approximation of PDF equations with closure. Implemented in PyTorch. 
# The reduced-order PDF equation is a deterministic advection-diffusion equation of the following form:
# 
#           p_t + d_dx ( G(t, x) * p ) = d2_dx2 ( D(t, x) * p)
# where G, D are respectively drift and diffusion coefficients that are generally depenedent on time. 


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
        super(F_Net, self).__init__()
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
        super(S_Net, self).__init__()
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

        # define LED closure data
        self.load_pde_physics()
        # load Fmc data computed in MATLAB
        self.load_Fmc_data()
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

    def build_models(self):
        """ Initialize neural nets. """
        # solution neural net
        self.F_nn = F_Net()
        self.S_nn = S_Net()

    def forward(self, inputs):
        """ 
            Generate approximate PDE solution. Input
            `X` should be grid data of form [x, t]. 
        """
        return self.F_nn(inputs)
    
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


    def load_data(self, path):
        """ Load and preprocess approximation on grid. """
        self.raw_data = scipy.io.loadmat("Fqoi.mat")
        # Fmc solution (Nx x Nt)
        self.Fmc = self.raw_data['Fqoi'] 
        # PDE domain
        self.xgrid = self.raw_data['x_grid'].reshape(-1, 1)
        self.tgrid = self.raw_data['t_grid'].reshape(-1, 1)
        self.xmin = self.xgrid.min()
        self.xmax = self.xgrid.max()
        self.tmin = self.tgrid.min()
        self.tmax = self.tgrid.max()
        # grid sizes (uniform grid)
        self.dt = self.tgrid[1]-self.tgrid[0]
        self.dx = self.xgrid[1]-self.xgrid[0]
        self.Nt = len(self.tgrid)
        self.Nx = len(self.xgrid)
        # total number of points in the domain
        self.N_grid = int(self.Nt*self.Nx)

    def load_pde_physics(self):
        """ 
            Loads advection, diffusion terms for the PDE, 
            and generates input grid points. 

            Input:
                (xmin, tmin)
                (x1, tmin),
                (x2, tmin),
                ...
                (xmax, tmin),
                ---
                (xmin, t1),
                (x1, t1),
                ...
                (xmax, t1),
                ...
                (xmin, tmax),
                ...
                (xmax, tmax)

            Output:
                Fnn(xmin, tmin),
                Fnn(x1, tmin),
                ...
                Fnn(xmax, tmin),
                Fnn(x1, t1),
                ...
                Fnn(x1, t1),
                ...
                Fnn(xmin, tmax),
                ...
                Fnn(xmax, tmax)
        """
        # PDE parameter: D_gam
        self.Dgam = self.raw_data['Dgam'][0][0]

        # <v(x,t)> = (th(x) - <h(t)>)/Dgam, stored as (Nx x Nt) matrix
        self.advection = self.raw_data['advection']

        # (\int_0^t<v'(t)v'(s)>ds)/Dgam^2, stored as (Nx x Nt) matrix (only depends on t, repeats in x axis)
        self.diffusion = self.raw_data['diffusion']

        # organize input data for PDE regression
        xgrid_pnts, tgrid_pnts = np.meshgrid(self.xgrid.reshape(-1, 1), self.tgrid.reshape(-1, 1))
        all_pnts = np.vstack([xgrid_pnts.ravel(), tgrid_pnts.ravel()]).T

        xr = torch.tensor(all_pnts[:, 0].reshape(-1, 1), requires_grad=True)
        tr = torch.tensor(all_pnts[:, 1].reshape(-1, 1), requires_grad=True)
        # all grid points for main domain PDE solution, should correspond to column-major flattening

        # size (Nx * Nt x 2)
        X_gridded = torch.concat([xr, tr], axis=1)

        # flatten advection term: (xi, tj) => <v(xi, tj)>
        advection_gridded = torch.tensor(self.advection.flatten("F").reshape(-1, 1))

        # flatten diffusion term: (xi, tj) => D(xi, tj), here `xi` is dummy variable
        diffusion_gridded = torch.tensor(self.diffusion.flatten("F").reshape(-1, 1))

        # !!!! todo: fit splines to advection and diffusion so random locations can be queried

        # bind data
        self.pde_data = (X_gridded, advection_gridded, diffusion_gridded)
    
    def load_Fmc_data(self):
        """ 
            loads solution data from numerical solver. 
            Flattened in column major.
            (xi, tj) => Fmc(xi, tj), where points are organized as:

            (xmin, tmin)
            (x1, tmin),
            (x2, tmin),
            ...
            (xmax, tmin),
            ---
            (xmin, t1),
            (x1, t1),
            ...
            (xmax, t1),
            ...
            (xmin, tmax),
            ...
            (xmax, tmax)
        """
        self.fmc_data = torch.tensor(self.Fmc.flatten("F")).reshape(-1, 1)

######################################################################
# Loss functions
######################################################################
    def domain_loss(self, inputs, outputs, advection_diffusion_data=None, fresh_samples=False):
        """ 
            PDE closure inside the domain, evaluated on random
            samples (independently generated of batch data).

            advection_diffusion_data:       specifies how the advection-diffusion
                                            terms are computed. By default, it is
                                            `None`, in which case the advection-
                                            diffusion terms will be interpolated
                                            using a pre-fitted spline at arbitrary
                                            points. Otherwise, the user can input 
                                            the discrete values at grid points, 
                                            organized in a (N x 2) array [adv, diff],
                                            please make sure that the queried indices 
                                            are consistent with those of the PDE solution
                                            `outputs`.

            fresh_samples:                  If `True`, randomly sample the domain and make
                                            predictions separately, instead of using the 
                                            input prediction data.
        """
        # !!! Note (01/05/2022) the advection and diffusion need to be redone 
        # as basis interpolations rather than grid points. Due to random batches,
        # it's difficult to keep track of the grid point indices (to index into
        # and get the correct advection and diffusion values.)

        if fresh_samples:
            raise NotImplementedError()
        # query advection and diffusion
        if advection_diffusion_data is None:
            raise NotImplementedError()
        else:
            advection = advection_diffusion_data[:, 0][:, None]
            diffusion = advection_diffusion_data[:, 1][:, None]

        # ----------------------------------------------------------------------
        # PDE Loss
        # ----------------------------------------------------------------------
        # predicted solution + forcing term
        F = outputs.clone()
        S = self.S_nn(inputs).clone()

        # partial derivatives
        dF = self.gradient(F, inputs, order=1)
        dF2 = self.gradient(F, inputs, order=2)
        dS = self.gradient(S, inputs, order=1)

        dFdx = dF[:, 0][:, None]
        dFdt = dF[:, 1][:, None]
        dF2dx2 = dF2[:, 0][:, None]
        dSdx = dS[:, 0][:, None]

        # assemble PDE loss
        lhs = (dFdt + (1/self.Dgam)*advection*dFdx - diffusion*dF2dx2)
        rhs = -(self.Dgam**2)*dSdx
        pde_loss = torch.mean((lhs - rhs)**2)

        # ----------------------------------------------------------------------
        # Monotinicity Loss at queried points
        # ----------------------------------------------------------------------
        #monotonicity_loss = torch.sum(( dFdx[dFdx < 0] )**2)/(self.Nx*self.Nt)
        monotonicity_loss = torch.sum(( dFdx[dFdx < 0] ).abs())

        return pde_loss, monotonicity_loss

    def data_loss(self, inputs, y_pred, y_true, gamma=0.0, ic_weight=10.0, bc_weight=20.0):
        """
            Evaluates data loss, or generalized least squares
            error (with `gamma`) between model predictions 
            and Monte Carlo data.

            Deviations on initional condition and boundary conditions 
            are penalized by weights `ic_weight`, `bc_weight`.
        """
        # number of points 
        b = len(y_pred)
        # generalized loss
        if gamma != 0:
            # need to be careful about 0/0
            residual = ( (y_pred - y_true)/(torch.abs(F)**gamma) )**2
        else:
            residual = (y_pred - y_true)**2
        
        initial_data_locations = (inputs[:, 1][:, None] == 0)
        left_boundary_data_locations = (inputs[:, 0][:, None] == self.xmin)
        right_boundary_data_locations = (inputs[:, 0][:, None] == self.xmax)
        # penalize initial condition
        residual *= torch.where(initial_data_locations, 
                                ic_weight*torch.ones_like(y_pred), 
                                torch.ones_like(y_pred))
        # penalize boundary condition (left boundary)
        residual *= torch.where(left_boundary_data_locations, 
                                bc_weight*torch.ones_like(y_pred), 
                                torch.ones_like(y_pred))
        # penalize boundary condition (right boundary)
        residual *= torch.where(right_boundary_data_locations, 
                                bc_weight*torch.ones_like(y_pred), 
                                torch.ones_like(y_pred))
        # categorize different loss terms
        if len(initial_data_locations) == 0:
            initial_loss = torch.tensor([0.0])
        else:
            initial_loss = torch.sum(residual[initial_data_locations])/b
        
        if len(left_boundary_data_locations) == 0:
            left_boundary_loss = torch.tensor([0.0])
        else:
            left_boundary_loss = torch.sum(residual[left_boundary_data_locations])/b
        
        if len(right_boundary_data_locations) == 0:
            right_boundary_loss = torch.tensor(0.0)
        else:
            right_boundary_loss = torch.sum(residual[right_boundary_data_locations])/b
        # total loss on data
        data_loss = torch.mean(residual)
        return data_loss, initial_loss, left_boundary_loss, right_boundary_loss

    def loss(self, y_pred, y_true, randomized=False, weighting=False):
        """ 
            Compute aggregate loss function from a 
            suite of loss functions.

            If loss `weighting` is not `False`, need to input a 
            vector of weights (not necessarily summing to 1.0).

            The ordering of losses should be [pde, data, monotonicity]
        """
        raise NotImplementedError()
        pde_loss, monotone_loss = self.domain_loss()
        data_loss, initial_loss, \
             left_boundary_loss, right_boundary_loss = self.data_loss(y_pred, y_true) 
        final_loss = pde_loss + data_loss + monotone_loss
        print(" Total loss = {} | PDE loss = {} | Data loss = {} | M loss = {} ".format(
                final_loss.item(), pde_loss.item(), data_loss.item(), 
                monotone_loss.item()
            ))
        return final_loss
    
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

if __name__ == "__main__":
    # debug
    pinn = PhysicsInformedROCDF(2, 1, Fmc_data_path="/Fqoi.mat")
    X = pinn.pde_data[0]
    y = pinn.fmc_data
    ########## debugging below

    