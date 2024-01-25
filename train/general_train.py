import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(os.curdir)))
from networks import network_factory
from data_loaders import get_train_loader


def train_neural_network(network_type, input_size, output_size, hidden_sizes):
    trainloader = get_train_loader()
    net = network_factory(network_type, input_size, output_size, hidden_sizes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(2):  # Change the number of epochs as needed
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, input_size)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    return net


if __name__ == "__main__":
    sys.path.insert(0, sys.path[-1])
    from test.test_accuracy import test_accuracy
    from config import input_size, output_size, hidden_sizes

    network_type = "fc"
    training_type = "E2ET"  # "gradual"
    str_sizes = '_'.join([str(s) for s in hidden_sizes])
    save_path = f"mnist_{network_type}_net_{training_type}_{str_sizes}.pth"
    network = train_neural_network(
        network_type, input_size, output_size, hidden_sizes
    )
    # from config import MODELS_DIR
    # torch.save(network.state_dict(), MODELS_DIR/save_path)
    torch.save(network.state_dict(), save_path)
    test_accuracy(network_type, training_type)
