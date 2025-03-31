from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from src.data import DATASET_ROOT


def load_fashion_mnist():
    root = os.path.join(DATASET_ROOT, "fashion_mnist")
    train_data = datasets.FashionMNIST(
        root=root, train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=root, train=False, download=True, transform=ToTensor()
    )

    return train_data, test_data


def load_cifar10():
    root = os.path.join(DATASET_ROOT, "cifar10")
    train_data = datasets.CIFAR10(
        root=root, train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root=root, train=False, download=True, transform=ToTensor()
    )

    return train_data, test_data


def load_cifar100():
    root = os.path.join(DATASET_ROOT, "cifar100")
    train_data = datasets.CIFAR100(
        root=root, train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.CIFAR100(
        root=root, train=False, download=True, transform=ToTensor()
    )

    return train_data, test_data
