import os
from pathlib import Path


input_size = 28 * 28  # MNIST image size
output_size = 10  # Number of classes in MNIST
hidden_sizes = [256, 256]  # List of hidden layer sizes

# enable to use export MNIST_DIR=/path/to/mnist/data/dir
print(f'env var: {os.environ.get("MNIST_DIR")}')
DATA_DIR = Path(os.environ.get("MNIST_DIR") or '../../data/mnist/mnist')
# MODELS_DIR = Path(os.environ.get("TRAINED_MODELS_DIR") or 'trained_models')

TRAINING_EPSILON = 0.05
