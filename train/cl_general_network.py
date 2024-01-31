"""
This file includes code that gradually trains neural network in a cascade 
learning fashion as follows:
in the i'th stage, it adds the i'th layer, and trains only the i'th layer
and the classification layer until convergence.
All other (first i-1) hidden layers are frozen.

The classification layer can be:
1) a duplicateion of the classification layer from the last stage or
2) a newly initialized classification layer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.curdir)))
from networks import network_factory
from data_loaders import get_train_loader
from config import input_size, output_size, hidden_sizes, TRAINING_EPSILON


def train_neural_network(
    network_type, input_size, output_size, hidden_sizes
):
    trainloader = get_train_loader()
    net = network_factory(
        network_type, input_size, output_size, hidden_sizes[0], 
        dup_or_init="dup"
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for layer_index,_ in enumerate(hidden_sizes):
        net.add_layer()
        # current optimizer optimizes only the new layer and the classifier
        num_layers = layer_index + 2
        optim_params = [
            {'params': net.model[2*i].parameters(), 'lr': 0.000}
            for i in range(num_layers)
        ]
        optimizer = optim.SGD(optim_params, lr=0.001, momentum=0.9)
        optimizer.param_groups[layer_index]['lr'] = 0.001
        optimizer.param_groups[-1]['lr'] = 0.001

        lrs_before = [pg['lr'] for pg in optimizer.param_groups]

        # train layer until convergence
        prev_loss = torch.inf
        epoch = 0
        while True:
            # break
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
            if running_loss >= prev_loss + TRAINING_EPSILON:
                break
            prev_loss = running_loss
            epoch += 1
        print(f'Finished Training layer {layer_index} in {epoch} epochs')
        # log learning rates update
        lrs_after = [pg['lr'] for pg in optimizer.param_groups]
        print(f'Learning rates update: {lrs_before} -> {lrs_after}')
    print('Finished Training')
    return net


if __name__ == "__main__":
    sys.path.insert(0, sys.path[-1])
    from test.test_accuracy import test_accuracy
    from config import input_size, output_size, hidden_sizes

    network_type = "fc_sc_clf"  # fully connected, skip connection, classifier
    training_type = "SC-CL"  # cascade learning, cascade learning
    str_sizes = '_'.join([str(s) for s in hidden_sizes])
    save_path = f"mnist_{network_type}_net_{training_type}_{str_sizes}.pth"
    network = train_neural_network(
        network_type, input_size, output_size, hidden_sizes
    )
    torch.save(network.state_dict(), save_path)
    test_accuracy(network_type, training_type)
