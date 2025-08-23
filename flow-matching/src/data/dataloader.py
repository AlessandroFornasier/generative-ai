import numpy as np
from torch.utils.data import Dataset, DataLoader
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


class LettersDataset(Dataset):
    def __init__(self, n_samples, std=0.1, seed=None):
        """
        Args:
            n_samples (int): Total number of samples.
            std (float): Standard deviation of the noise.
            seed (int, optional): Random seed.
        """
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
        self.n_samples = n_samples
        self.std = std
        
        a_points = []
        for t in np.linspace(0, 1, 20):
            a_points.append([-7.5 + t * 3.75, -7.5 + t * 15])
            a_points.append([-3.75 + t * 3.75, 7.5 - t * 15])
            a_points.append([-5.625 + t * 3.75, 0])
        
        f_points = []
        for t in np.linspace(0, 1, 20):
            f_points.append([0, -7.5 + t * 15])
            f_points.append([t * 7.5, 7.5])
            f_points.append([t * 7.5, 0])
        
        all_points = np.array(a_points + f_points, dtype=np.float32)
        
        n_template_points = len(all_points)
        indices = np.random.choice(n_template_points, size=n_samples, replace=True)
        self.data = all_points[indices] + np.random.normal(0, self.std, (n_samples, 2)).astype(np.float32)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

class MultimodalGaussianDataloader:
    """
    Factory class for multimodal Gaussian dataloader.
    
    Args:
        n_samples (int): Total number of samples.
        n_modes (int): Number of Gaussian modes.
        dim (int): Dimensionality of each sample.
        std (float): Standard deviation of each mode.
        seed (int, optional): Random seed for reproducibility.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
    """
    def __init__(self, n_samples, n_modes, dim=2, scale=1.0, std=0.1, seed=None, batch_size=32, shuffle=True) -> None: 
        self.dataset = MultimodalGaussianDataset(n_samples, n_modes, dim, scale, std, seed)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> DataLoader:
        """ Returns:
            DataLoader: DataLoader for the multimodal Gaussian dataset.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
class LettersDatasetDataloader:
    """
    Factory class for letters dataset dataloader.
    
    Args:
        n_samples (int): Total number of samples.
        seed (int, optional): Random seed for reproducibility.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
    """
    def __init__(self, n_samples, std=0.1, seed=None, batch_size=32, shuffle=True) -> None: 
        self.dataset = LettersDataset(n_samples, std, seed)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self) -> DataLoader:
        """ Returns:
            DataLoader: DataLoader for the letters dataset.
        """
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)