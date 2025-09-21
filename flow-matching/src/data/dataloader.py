import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import torch

class MultimodalGaussianDataset(Dataset):
    def __init__(self, n_samples, n_modes, dim=2, scale=1.0, std=0.1, seed=None):
        """
        Args:
            n_samples (int): Total number of samples.
            n_modes (int): Number of Gaussian modes.
            dim (int): Dimensionality of each sample.
            std (float): Standard deviation of each mode.
            seed (int, optional): Random seed.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
        self.n_samples = n_samples
        self.n_modes = n_modes
        self.dim = dim
        self.scale = scale
        self.std = std

        if dim == 2:
            angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
            self.centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        else:
            self.centers = np.random.randn(n_modes, dim)

        self.centers *= self.scale
        self.mode_assignments = np.random.choice(n_modes, size=n_samples)
        self.data = np.zeros((n_samples, dim), dtype=np.float32)
        for i in range(n_modes):
            idx = np.where(self.mode_assignments == i)[0]
            self.data[idx] = (
                self.centers[i] + np.random.randn(len(idx), dim) * std
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    
class MoonsDataset(Dataset):
    def __init__(self, n_samples, noise=0.0, seed=None):
        """
        Args:
            n_samples (int): Total number of samples.
            noise (float): Standard deviation of the Gaussian noise added to the data.
            seed (int, optional): Random seed.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
        data, _ = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]