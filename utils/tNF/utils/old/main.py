import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os

#from diffusioncharacterization.ctrw.random_walks import advection_diffusion_random_walk
from temporal_normalizing_flows.neural_flow import neural_flow
from temporal_normalizing_flows.latent_distributions import gaussian
from temporal_normalizing_flows.preprocessing import prepare_data

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass


# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# creating data set
def advection_diffusion_random_walk(walk_params, traj_params, initial_conditions):
    num_steps, num_walkers, dt = walk_params
    Diff, v = traj_params

    steps = np.random.normal(loc=v*dt, scale=np.sqrt(2*Diff*dt), size=(num_steps, num_walkers))
    trajectory = np.concatenate((initial_conditions, initial_conditions + np.cumsum(np.array(steps), axis=0)), axis=0)
    time = np.arange(num_steps + 1) * dt

    return time, trajectory

walk_params = [99, 500, 0.05]  # timesteps, walkers, stepsize
traj_params = [2.0, 2.5]       # Diffusion coefficient, velocity
initial_conditions = np.random.normal(loc=1.5, scale=0.5, size=(1, walk_params[1]))

time, position = advection_diffusion_random_walk(walk_params, traj_params, initial_conditions)

if __name__ == "__main__":
    ##########
    # training parameters
    num_iter = 1000

    model_path = "./trained_models/RandomWalkDiffusion_Advection_iter{}".format(num_iter)
    ##########
    x_sample = np.linspace(-15, 15, 1000)
    t_sample = time
    dataset = prepare_data(position, time, x_sample, t_sample)

    # create base model
    flow = neural_flow(gaussian)

    if os.path.isfile(model_path):
        # load trained parameters
        flow.load_state_dict(torch.load(model_path))
    else:
        # train model

        # ensure in the correct directory
        if os.path.isdir("./trained_models/"):
            flow.train(dataset, num_iter)
            torch.save(flow.state_dict(), model_path)
        
    # save trained model
    px, pz, jacob, z = flow.sample(dataset)
    plt.figure(1)
    plt.contourf(px); plt.xlabel("x"); plt.ylabel("t")
    plt.show()
    print(px.shape)
    plt.figure(2)
    plt.plot(x_sample, px[80, :])
    plt.show()














