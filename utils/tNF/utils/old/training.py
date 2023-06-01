"""
    Main routine for training the temporal Normalizing Flow
"""

# import pytorch
from tabnanny import check
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distrib
import torch.distributions.transforms as transform

# numpy
import numpy as np

import torch.optim as optim

# set number of threads (8 threads should work on most computers)
torch.set_num_threads(8)

import matplotlib.pyplot as plt

# supress warnings
import warnings
warnings.filterwarnings("ignore")


def train(dataset, model, log_base_measure, 
          num_epochs=200,
          batch_size=2**8,
          verbose=True,
          lr=1e-3, 
          use_scheduler=False,
          schedule_rate=0.9999,
          grad_clip=1e+4, 
          plot_it=True):
    """"
        Trains a normalizing flow model with time dependence
        to perform density estimation from particle trajectories.
        
        By default, the routine is minimizing negative log-likelihood.

    Args:
        dataset (ParticleDataset)        contains training and test data.

        model  (flow_model)
        base_measure (function)                specify base distribution of the normalizing flow.
                                         Ideally, the base distribution should coincide
                                         with the initial condition of Fokker Planck equation.
        dataloader (torch.utils.DataLoader) 
        num_epochs
        batch_size
        verbose (bool)
        lr (float): learning rate
        use_scheduler (bool): if learning rate schedule should be used, default to ExponentialScheduler
        step_schedule (int):  learning rate exponential decay momentum

        plot_it       (bool)             Plot the samples or not, must set to False
                                         if matplotlib is unavailable.
    
    """
    # customized negative log-likelihood loss function
    def loss_func(prior_logpdf, log_abs_det_jac, plotit=False):
        """
            Evaluates negative log-likelihood as given by the model.
            Given log-likelihoods of latent samples (obtained 
            from applying flow on observed samples) and associated 
            volume correction terms, computes estimated negative 
            log-likelihood of the observations given by the formula:

                L(w) = -sum_i(log p_Z(f(x_i)) + log abs det(df/dx)(x_i)) 


        Inputs:
            `z`         A batch of the data (after inverse flow), 
                        has size (b x (d+1)), assumed to contain 
                        the time dimenion:
                            z[i, :] = (t_i, z(t_i))
        """
        # sum log determinants across flow maps
        sum_log_det = sum(log_abs_det_jac)
        return -(prior_logpdf.reshape(-1, 1) + sum_log_det.reshape(-1, 1)).mean()
    
    # save hyperparameters before training
    initial_learning_rate = lr

    # initialize training model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, schedule_rate)
    
    # record preallocate
    all_training_loss = np.zeros([num_epochs])      # training loss averaged over each epoch
    all_test_loss = np.zeros([num_epochs])          # generalization loss computed on all test data at once
    model_snapshots = [model.state_dict()]          # snapshots of NF recorded each epoch
    all_grad_norms = np.zeros([num_epochs])         # record norm of gradients


    epoch = 0
    # number of allowance to redo one epoch
    patience = 0
    while epoch < num_epochs:

        # get torch dataloader for each epoch
        tensorizing_dataloader = dataset.get_dataloader(batch_size=batch_size)
        # accumulate loss over batch (used for computing average)
        acc_train_loss = []

        # save model, optimizer state for retraining
        save_model_state = model.state_dict()
        save_optimizer_state = optimizer.state_dict()

        # iterate over each batch
        for batch_number, minibatch in enumerate(tensorizing_dataloader):
            # observed samples before flowing
            x = minibatch

            # apply Normalizing Flow: f(x) => z
            z, log_jacobians = model(x.float())

            # evaluate latent log-density

            z_logpdf = log_base_measure(z)

            # clean optimizer for backward()
            optimizer.zero_grad()

            # compute minibatch sample KL divergence
            loss_train = loss_func(z_logpdf, log_jacobians)

            # backprop
            loss_train.backward()

            # check if training loss blew up, if it did, do not take step
            # namely that batch is discarded because it was "unlucky"
            if loss_train.item() <= 1e+16:
                # take a step

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                # update optimizer
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                # accumulate training loss per epoch
                acc_train_loss.append(loss_train.item())
                if verbose and (batch_number % 100 == 0):
                    print('[=> In training ...  ... (epoch{}=>batch{})'.format(epoch+1, batch_number))
            else:
                # do not take a step
                print('[=> Training encountered blowup: {} ...  ... (epoch{}=>batch{}) discarded! '\
                .format(loss_train.item(), \
                epoch+1, batch_number))
                continue
 
        ## Reporting after each epoch

        # get test data
        x_test = dataset.get_test_data()

        # use current NF model and flow all test samples
        z_test, log_jacobians_test = model(x_test.float())
        
        # evaluate latent log pdf
        z_test_logpdf = log_base_measure(z_test)


        # compute generalization err. on all test data at once
        loss_test = loss_func(z_test_logpdf, log_jacobians_test).item()

        # compute averaged training error
        acc_train_loss = np.mean(acc_train_loss)        

        # check for NaN loss value and stop training
        if np.isinf(loss_test) or np.isnan(loss_test) or np.isnan(acc_train_loss):
            print('Training stopped because loss became Inf or NaN!')

            return {'model_snapshots': model_snapshots, 
                    'training_loss': all_training_loss,
                    'test_loss': all_test_loss,
                    'grad_norms': all_grad_norms,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'lr': initial_learning_rate,
                    'post_training_samples': None,
                    'message': 'Training stopped because loss became Inf or NaN!'
                    }
        # should not do more than 20 times per epoch
        if patience >= 20:
            print('Terminating as patience is exceeded ... tune hyperparameters instead. ')
            return {'model_snapshots': model_snapshots, 
                    'training_loss': all_training_loss,
                    'test_loss': all_test_loss,
                    'grad_norms': all_grad_norms,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'lr': initial_learning_rate,
                    'post_training_samples': None,
                    'message': 'Terminating as patience is exceeded ... tune hyperparameters instead. '
                    }

        check_blow_up = True
        if check_blow_up:
            # check for test loss blowing up; should not blow up to more than initial loss
            if loss_test > 1e+16:
                print('Reverting (epoch{}) because test error blew up: {} ... '.format(epoch+1, loss_test))
                # revert model state back to before this epoch
                # implicitly, because we didn't save learning rate scheduler state
                # this means we retrain this epoch with a smaller learning rate
                model.load_state_dict(save_model_state)
                optimizer.load_state_dict(save_optimizer_state)
                optimizer.zero_grad()
                # increment patience
                patience += 1
                continue

        # record after epoch training
        all_training_loss[epoch] = acc_train_loss
        all_test_loss[epoch] = loss_test
        
        # save a snapshot of the model after each epoch
        model_snapshots.append(model.state_dict())

        # record norm of gradients as sanity check (prevent exploding gradient)
        total_grad_norm = np.sum([( p.grad.detach().data.norm(2) ) ** 2 for p in model.parameters() if p.grad is not None])
        if total_grad_norm <= 1e-6:
            break

        all_grad_norms[epoch] = total_grad_norm

        # report training progress
        if verbose:
            print("[ Epoch  {} => ( Train Avg. Over Epoch ) = {}, ( Generalization ) = {}"\
                .format(epoch + 1, acc_train_loss, loss_test))
            if use_scheduler:
                print('[=> Report Learning Rate =  {}'.format(optimizer.param_groups[0]['lr']))
            print('[=> Report Norm of Gradient =  {}'.format(total_grad_norm))
        
        # check next epoch
        epoch += 1
        # reset patience
        patience = 0

    # save all necessary reports
    to_return = {
        'model_snapshots': model_snapshots,
        'training_loss': all_training_loss,
        'test_loss':all_test_loss,
        'grad_norms': all_grad_norms,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': initial_learning_rate,
        'message': 'Success. '
    }
    return to_return