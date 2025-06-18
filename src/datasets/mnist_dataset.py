import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image

class MnistTransform:
    """
    Transform class for MNIST dataset.
    Applies a series of transformations to the images, including converting to tensor and normalizing.
    """
    def __init__(self):
        self.transforms = [
            self.to_grayscale,
            self.to_tensor,
            self.normalize
        ]

    def __call__(self, image, target=None):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target

    def to_grayscale(self, image, target):
        if image.mode != 'L':
            image = image.convert('L')
        return image, target

    def to_tensor(self, image, target):
        image = F.to_tensor(image)
        return image, target

    def normalize(self, image, target, mean=(0.1307,), std=(0.3081,)):
        image = F.normalize(image, mean=mean, std=std)
        return image, target

class MnistDataset(Dataset):
    """
    Custom dataset class for loading the MNIST dataset.
    This class extends the PyTorch Dataset class and provides methods to load data,
    apply transformations, and retrieve samples.
    """
    def __init__(self,
                 root: str = "data",
                 split: bool = True,
                 download: bool = True,
                 transform: MnistTransform = MnistTransform()):
        self.root = root
        self.split = split
        self.download = download
        self.data = self.load_data()
        self.transform = transform

    def load_data(self):
        data = datasets.MNIST(root=self.root,
                               train=self.split,
                               download=self.download,
                               transform=None)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, target = self.data[idx]

        if self.transform is not None:
            image, target = self.transform(image, target)

        return {
            'image': image,
            'target': target,
            'id': idx
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batching of data.
        Args:
            batch (list): List of samples from the dataset.
        Returns:
            tuple: A tuple containing a batch of images and their corresponding targets.
        """
        images = torch.stack([item['image'] for item in batch])        # [B, C, H, W]
        targets = torch.tensor([item['target'] for item in batch])     # [B]
        ids = [item['id'] for item in batch]

        return {
            'images': images,
            'targets': targets,
            'ids': ids
        }
