import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

class MNISTDataLoader(DataLoader):
    """
    MNIST DataLoader class.
    """
    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

    def get_dataloader(self, data_path: str, train: bool = True) -> DataLoader:
        dataset = datasets.MNIST(root=data_path, train=train, download=True, transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)