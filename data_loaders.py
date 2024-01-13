import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from os.path import exists
from config import DATA_DIR


def get_train_loader():
    # MNIST data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dir = DATA_DIR  # / "mnist"
    print(f"DATA_DIR={DATA_DIR}")
    do_download = not exists(mnist_dir)
    trainset = datasets.MNIST(root=mnist_dir, train=True, download=do_download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    return trainloader


def get_testloader():
    # Download the MNIST test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    return testloader
