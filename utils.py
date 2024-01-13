import torch
import numpy as np
from networks import network_factory

def load_and_run(net_path, input_vector=None):
    # code to load the network:
    net = network_factory("fc", 784, 10, [128,64])
    if input_vector is None:
        input_vector = torch.Tensor(np.array([0.1]*784))
    output_vector = net(input_vector)
    print(output_vector)

load_and_run(net_path = "mnist_fc_net_E2ET_256_256.pth")
