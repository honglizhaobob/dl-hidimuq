# Utility functions shared across all NN models

# Author: Hongli Zhao
# Date: 07/12/2023

##########################################################################################
# list of imports
##########################################################################################
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

##########################################################################################
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

######################################################################
# Training routine
######################################################################
def train(
        model, optim, scheduler,
        batch_size, epochs=50, 
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
    all_epoch_reg_loss = []
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
            X, y = X[tmp, :].data, y[tmp, :].data # the `.data` detaches computational graph

        # loop over batches
        # ----------------------------------------
        # Define batch-wise variables
        #
        all_batch_losses = []
        all_batch_losses_data = []
        all_batch_losses_pde = []
        all_batch_losses_init = []
        all_batch_losses_boundary = []
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
                    pde_loss = model.domain_loss()

                    # no data regularization enforced
                    data_loss = torch.tensor(0.0)

                else:
                    raise NotImplementedError()

                # ----------------------------------------
                #          Regularity loss
                # ----------------------------------------
                # computed separately by querying time points
                
                # query on random points
                reg_loss = model.regularity_loss(n_query=20)
                
                # aggregate (pde_loss + initial_loss + boundary_loss) + data_loss + reg_loss
                train_loss = pde_loss + data_loss + reg_loss
                # ----------------------------------------
                #          Save history
                # ----------------------------------------
                all_batch_losses_pde.append(pde_loss.item())
                all_batch_losses_data.append(data_loss.item())
                all_batch_losses_reg.append(reg_loss.item())
                all_batch_losses.append(train_loss.item())

                #all_batch_losses_init.append(initial_loss.item())
                #all_batch_losses_boundary.append(boundary_loss.item())

                # compute backpropagation
                train_loss.backward()
                return train_loss
            
            optim.step(closure=closure)

        # ----------
        if scheduler:
            # step scheduler after epoch if there is one
            scheduler.step()

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
            mean_batch_pde_loss = np.mean(all_batch_losses_pde)
            #mean_batch_init_loss = np.mean(all_batch_losses_init)
            #mean_batch_boundary_loss = np.mean(all_batch_losses_boundary)
            print("                     P Loss             = {}".format(mean_batch_pde_loss))# + mean_batch_init_loss + mean_batch_boundary_loss))
            print("                     |    domain        = {}".format(mean_batch_pde_loss))
            #print("                     |    init          = {}".format(mean_batch_init_loss))
            #print("                     |    bound         = {}".format(mean_batch_boundary_loss))
            # save 
            #all_epoch_init_loss.append(mean_batch_init_loss)
            #all_epoch_boundary_loss.append(mean_batch_boundary_loss)
            all_epoch_pde_loss.append(mean_batch_pde_loss)
        if len(all_batch_losses_data) > 0:
            mean_batch_data_loss = np.mean(all_batch_losses_data)
            print("                     D Loss             = {}".format(mean_batch_data_loss))
            # save
            all_epoch_data_loss.append(mean_batch_data_loss)

        if len(all_batch_losses_reg) > 0:
            mean_batch_reg_loss = np.mean(all_batch_losses_reg)
            print("                     R Loss             = {}".format(mean_batch_reg_loss))
            # save
            all_epoch_reg_loss.append(mean_batch_reg_loss)
        

    # save info and return 
    info = {
        "pde_loss": all_epoch_pde_loss, 
        "data_loss": all_epoch_data_loss,
        "reg_loss": all_epoch_reg_loss
        #"init_loss": all_epoch_init_loss,
        #"boundary_loss": all_epoch_boundary_loss
    }
    return info