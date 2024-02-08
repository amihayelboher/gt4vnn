import torch

import torch.nn as nn
import sys, os
sys.path.append(os.path.abspath(os.curdir))
from config import input_size, output_size, hidden_sizes
from networks import network_factory
from data_loaders import get_testloader


network_type = "fc_sc_clf"
training_type = "SC-CL"
str_sizes = '_'.join([str(s) for s in hidden_sizes])
model_path = f"mnist_{network_type}_net_{training_type}_{str_sizes}.pth"
model = network_factory(network_type, input_size, output_size, hidden_sizes)
model.load_state_dict(torch.load(model_path))
model.eval()
model.active_layers = len(model.layers)

testloader = get_testloader()

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images.view(images.size(0), -1))
        break

out0_clf1 = model.classifiers[1](model.intermediate_results[0])
out0_clf0 = model.classifiers[0](model.intermediate_results[0])
out1_clf1 = model.classifiers[1](model.intermediate_results[1])
out1_clf0 = model.classifiers[0](model.intermediate_results[1])
out01_clf1 = model.classifiers[1](model.intermediate_results[0] + model.intermediate_results[1])
out01_clf0 = model.classifiers[0](model.intermediate_results[0] + model.intermediate_results[1])

print((model.intermediate_results[0] == model.intermediate_results[1]).all())

print(f"out0_clf0: {[out0_clf0[i].argmax() for i in range(images.shape[0])]}")
# >> [tensor(7), tensor(2), tensor(1), tensor(0)]
print(f"out0_clf1: {[out0_clf1[i].argmax() for i in range(images.shape[0])]}")
# >> out0_clf1: [tensor(7), tensor(2), tensor(8), tensor(0)]
print(f"out1_clf0: {[out1_clf0[i].argmax() for i in range(images.shape[0])]}")
# >> out1_clf0: [tensor(8), tensor(4), tensor(4), tensor(6)]
print(f"out1_clf1: {[out1_clf1[i].argmax() for i in range(images.shape[0])]}")
# >> out1_clf1: [tensor(8), tensor(2), tensor(4), tensor(4)]
print(f"out01_clf1: {[out01_clf1[i].argmax() for i in range(images.shape[0])]}")
# >> out01_clf1: [tensor(7), tensor(2), tensor(8), tensor(0)]
print(f"out01_clf0: {[out01_clf0[i].argmax() for i in range(images.shape[0])]}")
# >> out01_clf0: [tensor(7), tensor(2), tensor(1), tensor(0)]
print(f"outputs: {[outputs[i].argmax() for i in range(images.shape[0])]}")
# >> [tensor(7), tensor(2), tensor(1), tensor(0)]
