import torch
import os

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from src.utils import get_dataset_root

DATASET_ROOT = get_dataset_root()

def load_fashion_mnist(transform):
    root = os.path.join(DATASET_ROOT, "fashion_mnist")
    train_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )

    test_data = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    return train_data, test_data


def load_cifar10(transform):
    root = os.path.join(DATASET_ROOT, "cifar10")
    train_data = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )

    test_data = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )

    return train_data, test_data


def load_cifar100(transform):
    root = os.path.join(DATASET_ROOT, "cifar100")
    train_data = datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform
    )

    test_data = datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform
    )

    return train_data, test_data

DATSET_LOADER = {
    "fashion_mnist": load_fashion_mnist,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
}


def create_image_classification_dataloader(dataset_name, batch_size, is_train=False):
    shuffle = True if is_train else False
    transform = Compose(
        [Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] 
    )
    if is_train:
        dataset = DATSET_LOADER[dataset_name](transform)[0]
    else:
        dataset = DATSET_LOADER[dataset_name](transform)[1]
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return dataloader