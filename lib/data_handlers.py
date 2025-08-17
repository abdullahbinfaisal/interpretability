import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import random_split


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




def Load_ImageNet100(root_dir=r"C:\Users\sproj_ha\Desktop\vision_interp\datasets\imagenet100", train=True, batch_size=64, shuffle=False, transform=None, dataset_allow=False):
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
    
    if train:
        root_dir = os.path.join(root_dir, "train.X1")
    else:
        root_dir = os.path.join(root_dir, "val.X")


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
    
    if not dataset_allow:
        return dataloader
    else:
        return dataloader, dataset



def Load_ImageNet100Sketch(root_dir=r"C:\Users\sproj_ha\Desktop\vision_interp\datasets\imagenetsketch\sketch", train=True, batch_size=64, shuffle=False, transform=None):
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
    
    if train == False:
        train_size = int(0.9 * len(dataset))  # 90% train
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader


class PACSAllDomains(Dataset):
    def __init__(self, root_dir, domains=None, transform=None):
        """
        Loads PACS dataset from all specified domains.

        Args:
            root_dir (str): Root directory of PACS dataset.
            domains (list of str, optional): List of domains to include. 
                                             Defaults to all 4 PACS domains.
            transform (callable, optional): Transform to apply to images.
        """
        if domains is None:
            domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = set()

        # Collect samples from all domains
        for domain in domains:
            domain_dir = os.path.join(root_dir, domain)
            for class_name in sorted(entry.name for entry in os.scandir(domain_dir) if entry.is_dir()):
                self.classes.add(class_name)
                class_dir = os.path.join(domain_dir, class_name)
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, file_name), class_name))

        # Sort classes and build index mapping
        self.classes = sorted(self.classes)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Map string class labels to integer indices
        self.samples = [(path, self.class_to_idx[label]) for path, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def Load_PACS(root_dir="./datasets/PACS", batch_size=64, shuffle=True, transform=None):
    """
    Loads the PACS dataset across all domains and returns a DataLoader.

    Args:
        root_dir (str): Path to the root directory of the PACS dataset.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle the dataset.
        transform (callable, optional): Image transform.

    Returns:
        DataLoader: PyTorch DataLoader for PACS across all domains.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    dataset = PACSAllDomains(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
