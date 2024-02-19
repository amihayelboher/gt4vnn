import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEED = 2024
DATASET = "MNIST"
# enable to use export MNIST_DIR=/path/to/mnist/data/dir
print(f'env var: {os.environ.get("MNIST_DIR")}')
print(f'env var: {os.environ.get("CIFAR10_DIR")}')

if DATASET == "MNIST":
    input_size = 28 * 28  # MNIST image size
    output_size = 10  # Number of classes in MNIST
    DATA_DIR = Path(os.environ.get("MNIST_DIR") or '../../data/mnist/mnist')
    # EPOCH2LOSS = {0:100, 1:20, 2:10, 3:5, 4:2, 5:1, 6:0.5}
    EPOCH2LOSS = {0:10, 1:5, 2:2, 3:1, 4:0.5, 5:0.25, 6:0.125}
elif DATASET == "CIFAR10":
    input_size = 3 * 32 * 32  # CIFAR10 image size
    output_size = 10  # Number of classes in CIFAR10
    EPOCH2LOSS = {0:25, 1:20, 2:30, 3:20, 4:15, 5:10}
    DATA_DIR = Path(os.environ.get("CIFAR10_DIR"))
else:
    raise Exception(f"Unknown dataset: {DATASET}")
hidden_sizes = [256, 256, 256, 256, 256, 256]  # List of hidden layer sizes
hidden_sizes = [256, 256]  # List of hidden layer sizes
# hidden_sizes = [50, 50, 50, 50]  # List of hidden layer sizes

# MODELS_DIR = Path(os.environ.get("TRAINED_MODELS_DIR") or 'trained_models')
