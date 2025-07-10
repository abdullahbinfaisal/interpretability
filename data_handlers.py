import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageNet100(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.class_to_idx = {}
        self.classes = sorted(entry.name for entry in os.scandir(root_dir) if entry.is_dir())
        
        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, file_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def Load_CIFAR(train = True, batch_size = 64, shuffle: bool = True, transform=None):
    """
    Loads the CIFAR-10 dataset and returns a DataLoader.

    Args:
        train (bool): If True, loads training data. If False, loads test data.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        DataLoader: PyTorch DataLoader for CIFAR-10.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    dataset = datasets.CIFAR10(
        root='./datasets',
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def Load_MNIST(train = True, batch_size = 64, shuffle = True, transform=None):
    """
    Loads the MNIST dataset and returns a DataLoader.

    Args:
        train (bool): If True, loads training data. If False, loads test data.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to apply to the data.

    Returns:
        DataLoader: PyTorch DataLoader for MNIST.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalization
        ])

    dataset = datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader




def Load_ImageNet100(root_dir="C:/Users/sproj_ha/Desktop/vision_interp/datasets/imagenet100/train.X1/", train=True, batch_size=64, shuffle=False, transform=None):
    """
    Returns a DataLoader for the custom ImageNet-100 dataset.

    Args:
        root_dir (str): Path to dataset root.
        train (bool): Whether to load training or test split.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle the dataset.
        transform (callable, optional): Image transform.

    Returns:
        DataLoader: PyTorch DataLoader for ImageNet-100.
    """
    
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])

    dataset = ImageNet100(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    

    return dataloader
