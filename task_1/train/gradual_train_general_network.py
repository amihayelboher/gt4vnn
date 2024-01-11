import torch
import torch.nn as nn
from os.path import exists
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.curdir)))
from task_1.task_1_utils import FullyConnected
from task_1.task_1_config import input_size, output_size, hidden_sizes, DATA_DIR


def train_neural_network(hidden_sizes):
    # MNIST data loading and preprocessing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dir = "/home/azencot_group/datasets/mnist"#DATA_DIR  # / "mnist"
    do_download = not exists(mnist_dir)
    trainset = torchvision.datasets.MNIST(root=mnist_dir, train=True, download=do_download, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    net = FullyConnected(input_size, output_size, hidden_sizes)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optim_parameters_list = [
        [{'params': net.model[0].parameters(), 'lr': 0.001},
         {'params': net.model[2].parameters(), 'lr': 0.0},
         {'params': net.model[4].parameters(), 'lr': 0.0},],
        [{'params': net.model[0].parameters(), 'lr': 0.0},
         {'params': net.model[2].parameters(), 'lr': 0.001},
         {'params': net.model[4].parameters(), 'lr': 0.001},]
    ]
        # [{'params': net.model[0].parameters(), 'lr': 0.0},
        #  {'params': net.model[2].parameters(), 'lr': 0.0},
        #  {'params': net.model[4].parameters(), 'lr': 0.001},]
    # ]
    optimizers = [optim.SGD(op, lr=0.001, momentum=0.9) for op in optim_parameters_list]

    # Training loop
    opt_index = 0
    for epoch in range(2):  # Change the number of epochs as needed
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.view(-1, input_size)
            optimizers[opt_index].zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[opt_index].step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # Print every 2000 batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        if epoch == 0:
            lrs_before = [pg['lr'] for pg in optimizers[opt_index].param_groups]
            opt_index += 1
            # optimizers[opt_index].param_groups[1]['lr'] = 0.001
            lrs_after = [pg['lr'] for pg in optimizers[opt_index].param_groups]
            print(f'Learning rates update: {lrs_before} -> {lrs_after}')
    print('Finished Training')
    return net


if __name__ == "__main__":
    save_path = f"mnist_net_GEN-GT_{'_'.join([str(s) for s in hidden_sizes])}.pth"
    # Example usage with hidden layer sizes [128, 64]:
    network = train_neural_network(hidden_sizes=hidden_sizes)
    torch.save(network.state_dict(), save_path)
