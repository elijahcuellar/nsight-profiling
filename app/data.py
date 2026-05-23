import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


def get_datasets():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    )

    return training_data, test_data


def get_dataloaders(training_data=None, test_data=None, batch_size: int = 64):
    """Return dataloaders for the provided datasets.

    If `training_data` or `test_data` are None, they will be downloaded by
    calling `get_datasets()`.
    """
    if training_data is None or test_data is None:
        training_data, test_data = get_datasets()

    train_loader = DataLoader(training_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader


def get_test_sample(dataset, idx: int = 0):
    return dataset[idx]
