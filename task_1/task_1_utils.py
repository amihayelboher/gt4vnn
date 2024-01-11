import torch.nn as nn

# Define the neural network
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

# code to load the network:
# net_path = "mnist_net_general_128_64.pth"
# net = NeuralNetwork(784,10,[128,64])
# net.forward(torch.Tensor(np.array([0.1]*784)))
