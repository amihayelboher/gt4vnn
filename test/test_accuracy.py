import torch

import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.curdir))
from config import input_size, output_size, hidden_sizes
from networks import network_factory
from data_loaders import get_testloader


def test_accuracy(network_type, training_type):
    str_sizes = '_'.join([str(s) for s in hidden_sizes])
    model_path = f"mnist_{network_type}_net_{training_type}_{str_sizes}.pth"
    model = network_factory(network_type, input_size, output_size, hidden_sizes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    testloader = get_testloader()
    # Evaluate the accuracy on the test set
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.view(images.size(0), -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print('Accuracy on the MNIST test set: {:.2%}'.format(accuracy))
    return accuracy


if __name__ == "__main__":
    network_type = "fc"
    training_type = "E2ET"  # "gradual"
    test_accuracy(network_type, training_type)
