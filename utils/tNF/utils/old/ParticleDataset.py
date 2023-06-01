
""" 
    Utility functions for simulating particle data and creating
    batched PyTorch datasets.

"""

# import pytorch dataloader
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import scipy
import scipy.io
import numpy as np

# supress warnings
import warnings
warnings.filterwarnings("ignore")

## Dataset classes for PyTorch training
class ParticleDataset(Dataset):
    """ 
        Constructs the dataset by simulating solutions
        of an input stochastic dynamical system.
        
    """
    def __init__(self, matfile, root_dir=os.getcwd() + "/data/"):
        if root_dir is not None:
            self.root_dir = root_dir
            # checks sample folder exists
            assert os.path.isdir(self.root_dir), "Please make sure sample folder is created. "
            # check if data is already simulated
            data_path = root_dir + matfile
            # load data
            self.raw_dataset = scipy.io.loadmat(data_path)
        else:
            assert isinstance(matfile, dict)
            assert "X_train" in matfile.keys()
            assert "X_test" in matfile.keys()
            self.raw_dataset = matfile

        # load samples
        self.training_data = self.raw_dataset['X_train']
        self.test_data = self.raw_dataset['X_test']

        # convert to torch.Tensor (requires grad optional)
        self.training_data = torch.tensor(self.training_data)
        self.test_data = torch.tensor(self.test_data)
        
        # hyperparameters
        self.num_samples = self.training_data.shape[0]
        # spatial dimension
        self.dim = self.training_data.shape[1] - 1

    def __len__(self):
        """ return dataset size. """
        return self.num_samples
    
    def __getitem__(self, idx):
        """ slice into data based on indices in IDX. """
        samples = self.training_data[idx, :]
        return samples

    def get_test_data(self):
        """ query all test data for validation. """
        return self.test_data

    def get_dataloader(self, batch_size, shuffle=False, num_workers=4, drop_last=True):
        """ Creates a PyTorch DataLoader with preferred parameters.
        This function is a wrapper. See: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        for a detailed explanation of parameters. 
        """
        if torch.cuda.is_available():
            # use cuda generator
            generator = torch.Generator(device="cuda")
        else:
            generator = torch.Generator(device="cpu")
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
            drop_last=drop_last, generator=generator)