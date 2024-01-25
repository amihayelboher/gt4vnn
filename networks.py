import torch.nn as nn
from copy import deepcopy

class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(FullyConnected, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes)-2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FullyConnectedSkipConnection(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=256, dup_or_init="dup"
    ):
        super(FullyConnectedSkipConnection, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        err_msg = f"invalid dup_or_init value: {dup_or_init}"
        assert dup_or_init in ["dup", "init"], err_msg
        self.dup_or_init = dup_or_init
        self.model = None
    
    def add_layer(self):
        if self.model is None:
            self.model = nn.Sequential(
                nn.Linear((self.input_size, self.hidden_size)),
                nn.ReLU(),
                nn.Linear((self.hidden_size, self.output_size)),
            )
        else:
            # get the current layers
            layers = [l for l in self.model]
            new_layer = nn.Linear((self.hidden_size, self.hidden_size))
            # add the new layer
            if self.dup_or_init == "dup":
                last_layer = deepcopy(layers[-1])
            else:  # self.dup_or_init == "init":
                last_layer = nn.Linear((self.hidden_size, self.output_size))
            layers = layers[:-1] + [new_layer, nn.ReLU(), last_layer]
            # create model
            self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


network_type2network_class = {
    "fc": FullyConnected, 
    "fc_sc_clf": FullyConnectedSkipConnection
}

def network_factory(network_type, input_size, output_size, hidden_sizes):
    network_class = network_type2network_class[network_type]
    return network_class(input_size, output_size, hidden_sizes)

# code to load the network:
# net_path = "mnist_net_general_128_64.pth"
# net = network_factory("fc", 784, 10, [128,64])
# net.forward(torch.Tensor(np.array([0.1]*784)))
