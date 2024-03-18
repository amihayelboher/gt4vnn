import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from os.path import exists
from config import DATA_DIR, DATASET, BATCH_SIZE
from torchvision.datasets import CIFAR10


def get_train_loader(dataset=DATASET):
    # MNIST data loading and preprocessing
    mnist_dir = DATA_DIR  # / "mnist"
    print(f"DATA_DIR={DATA_DIR}")
    do_download = not exists(mnist_dir)
    if dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(root=mnist_dir, train=True, download=do_download, transform=transform)
    elif dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(), 
            # Normalize to the range [-1, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
        ])
        trainset = datasets.CIFAR10(root=mnist_dir, train=True, download=do_download, transform=transform)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return train_loader


def get_testloader(dataset=DATASET):
    # Download the MNIST test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset == "MNIST":
        testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    elif dataset == "CIFAR10":
        testset = CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    return testloader
