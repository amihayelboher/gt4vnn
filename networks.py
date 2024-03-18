import torch.nn as nn
from copy import deepcopy


# if true, then initialize the layers randomly, and forward input through all layers
RAND_INIT_AND_FORWARD = True

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


# class FullyConnectedSkipConnection(nn.Module):
#     def __init__(
#         self, input_size, output_size, hidden_size=256, dup_or_init="dup"
#     ):
#         super(FullyConnectedSkipConnection, self).__init__()
#         self.input_size = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         err_msg = f"invalid dup_or_init value: {dup_or_init}"
#         assert dup_or_init in ["dup", "init"], err_msg
#         self.dup_or_init = dup_or_init
#         self.model = None
#         self.classifiers = []  # classifiers (last layers) during the training
    
#     def add_layer(self):
#         if self.model is None:
#             clf = nn.Linear(self.hidden_size, self.output_size)
#             self.model = nn.Sequential(
#                 nn.Linear(self.input_size, self.hidden_size),
#                 nn.ReLU(),
#                 clf,
#             )
#             self.classifiers.append(clf)
#         else:
#             # get the current layers
#             layers = [l for l in self.model]
#             new_layer = nn.Linear(self.hidden_size, self.hidden_size)
#             # add the new layer
#             if self.dup_or_init == "dup":
#                 last_layer = deepcopy(layers[-1])
#             else:  # self.dup_or_init == "init":
#                 last_layer = nn.Linear(self.hidden_size, self.output_size)
#             layers = layers[:-1] + [new_layer, nn.ReLU(), last_layer]
#             # create model
#             self.model = nn.Sequential(*layers)
#             self.classifiers.append(last_layer)

#     def forward(self, x):
#         return self.model(x)

class FullyConnectedSkipConnection(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_sizes=[256], dup_or_init="dup"
    ):
        super(FullyConnectedSkipConnection, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_size = hidden_sizes[0]
        err_msg = f"invalid dup_or_init value: {dup_or_init}"
        assert dup_or_init in ["dup", "init"], err_msg
        self.dup_or_init = dup_or_init
        self.layers = []
        self.classifiers = []  # classifiers (last layers) during the training
        for _ in range(len(hidden_sizes)):
            self.add_layer()
        self.layers = nn.ModuleList(self.layers)
        self.classifiers = nn.ModuleList(self.classifiers)
        self.active_layers = 1  # linear layers participating in forward
        self.intermediate_results = {}  # last forward() intermediate results
        self._register_hooks()

    def _register_hooks(self):
        # capturing intermediate results by registering forwars pass hooks
        def hook_fn(module, input, output, layer_idx):
            self.intermediate_results[layer_idx] = output
        hooks = []
        for layer_idx,layer in enumerate(self.layers):
            hook = layer.register_forward_hook(
                lambda module, input, output, 
                layer_idx=layer_idx: hook_fn(module, input, output, layer_idx)
            )
            hooks.append(hook)
        self._hooks = hooks

    def add_layer(self):
        if not self.layers:
            self.layers.extend([
                nn.Linear(self.input_size, self.hidden_size),
            ])
            clf = nn.Linear(self.hidden_size, self.output_size)
            self.classifiers.append(clf)
        else:
            new_layer = nn.Linear(self.hidden_size, self.hidden_size)
            self.layers.append(new_layer)
            last_layer = nn.Linear(self.hidden_size, self.output_size)
            if self.dup_or_init == "dup":
                nn.init.zeros_(last_layer.weight)
                nn.init.zeros_(last_layer.bias)
            self.classifiers.append(last_layer)

    def forward(self, x):
        self.intermediate_results = {}  # Clear previous results
        forward_layers = self.active_layers
        if RAND_INIT_AND_FORWARD:
            forward_layers = len(self.layers)
        for i in range(forward_layers):
            x = nn.functional.relu(self.layers[i](x))
        sum_ims = self.intermediate_results.values()
        return self.classifiers[self.active_layers-1](sum(sum_ims))
    
    # def get_forward_ratios(self, x, all_ratios=True):
    #     # after loading a network active_layers = 0, so we change it by defualt
    #     if all_ratios:  # change active_layers to get ratio for each layer
    #         orig_active_layers = self.active_layers
    #         self.active_layers = len(self.layers)
    #     output = self.forward(x)
    #     winner = output.argmax()
    #     classifier = self.classifiers[self.active_layers-1]
    #     im_preds = [classifier(ir)[winner] for ir in self.intermediate_results]
    #     if all_ratios:  # restore self.active_layers
    #         self.active_layers = orig_active_layers
    #     assert sum(im_preds) == output[winner]
    #     return [ip/sum(im_preds) for ip in im_preds]


    def activate_next_layer(self):
        # don't activate if all layers are activated
        if self.active_layers >= len(self.hidden_sizes):
            return
        if self.dup_or_init == "dup":
            # duplicate by adding current clf weights to zero weights
            cur_clf = self.classifiers[self.active_layers-1]
            print(f"len(self.classifiers)={len(self.classifiers)}")
            print(f"self.active_layers={self.active_layers}")
            next_clf = self.classifiers[self.active_layers]
            next_clf.weight.data += cur_clf.weight.data.detach()
            next_clf.bias.data += cur_clf.bias.data.detach()
        self.active_layers += 1


class SequentialWithSkipConnection(nn.Module):
    """
    a sequential network that is used to ease the translation from 
    FullyConnectedSkipConnection (from now on FCSC) to onnx.
    __init__ gets a triple (FCSC network, number of layers, classifier index),
    and generate a sequential network with the relevant layers and clasifier.
    default value for the number of layers is the number of active layers.
    default value for classifier index is -1 (last classifier).
    """
    def __init__(self, fcsc_net, num_of_layers, clf_index):
        super(SequentialWithSkipConnection, self).__init__()
        self.input_size = fcsc_net.input_size
        self.output_size = fcsc_net.output_size
        self.hidden_sizes = fcsc_net.hidden_sizes
        self.hidden_size = fcsc_net.hidden_sizes[0]
        if clf_index is None:
            clf_index = -1
        if num_of_layers is None:
            num_of_layers = fcsc_net.active_layers
        self.model = nn.Sequential(
            *fcsc_net.layers[:num_of_layers],
            fcsc_net.classifiers[clf_index]
        )

    def forward(self, x):
        cum_sum = 0.0
        for layer in self.model[:-1]:
            x = layer(x)
            cum_sum += x
        return self.model[-1](cum_sum)


network_type2network_class = {
    "fc": FullyConnected, 
    "fc_sc_clf": FullyConnectedSkipConnection
}

def network_factory(
    network_type, input_size, output_size, hidden_sizes, **kws
):
    network_class = network_type2network_class[network_type]
    return network_class(input_size, output_size, hidden_sizes, **kws)

# code to load the network:
# net_path = "mnist_net_general_128_64.pth"
# net = network_factory("fc", 784, 10, [128,64])
# net.forward(torch.Tensor(np.array([0.1]*784)))
