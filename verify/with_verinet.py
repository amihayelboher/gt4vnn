# NOTE: VeriNet only supports networks with a single output layer
# trying to validate other networks results with:
# ValueError: VeriNet only supports networks with a single output layer (with connections_to = [])

import sys
sys.path.append("/home/yizhak/Research/Code/VeriNet/")
sys.path.append("/home/yizhak/Research/Code/gt4vnn/")
import torch
import numpy as np
from verinet.parsers.onnx_parser import ONNXParser
from verinet.verification.objective import Objective
from verinet.verification.verinet import VeriNet
from verify.verify_utils import get_trained_network

torch_path = "/home/yizhak/Research/Code/gt4vnn/mnist_fc_sc_clf_net_SC-CL_256_256.pth"
tnet, onet = get_trained_network(torch_path, validate=True, partial=False)

import torch.nn as nn
from verinet.neural_networks.verinet_nn import VeriNetNN, VeriNetNNNode
from config import input_size, output_size, hidden_sizes


nodes = [VeriNetNNNode(idx=0, op=nn.Identity(), connections_from=None, connections_to=[1])]
# for each couple of current+next hidden-sizes, add layer of shape [current,next]
from_sizes = [input_size] + hidden_sizes[:-2]  
to_sizes = hidden_sizes[1:]
for i,(s1,s2) in enumerate(zip(from_sizes, to_sizes)):
    # add 2 layers: Linear and Relu
    nodes.append(
        VeriNetNNNode(
            idx=2*i+1, op=nn.Linear(s1,s2), 
            connections_from=[2*i], connections_to=[2*(i+1)]
        ),
    )
    nodes.append(
        VeriNetNNNode(
            idx=2*(i+1), op=nn.ReLU(), 
            connections_from=[2*i+1], connections_to=[2*(i+1)+1]
            ),
    )
VeriNetNNNode(idx=2*(i+1), op=nn.Linear(hidden_sizes[-1], output_size), connections_from=[2*i+1], connections_to=[2*(i+1)+1])
# VeriNetNNNode(idx=3, op=nn.Identity(), connections_from=[2], connections_to=None)

model = VeriNetNN(nodes)
# onnx_parser = ONNXParser(onnx_model_path, input_names=("x",), transpose_fc_weights=False, use_64bit=False)
# model = onnx_parser.to_pytorch()
model.eval()

# local robustness (no adversarial examples)
x = np.array(list([0.1]*784))
DELTA = 0.0005  # SAT
input_bounds = np.concatenate([(x-DELTA)[:,np.newaxis], (x+DELTA)[:, np.newaxis]], axis=1)
y = tnet.forward(torch.Tensor(x)).argmax()
objective = Objective(input_bounds, output_size=10, model=model)
out_vars = objective.output_vars
for j in range(objective.output_size):
    if j != y:
        # noinspection PyTypeChecker
        objective.add_constraints(out_vars[j] <= out_vars[y])

solver = VeriNet(use_gpu=False, max_procs=None)
status = solver.verify(objective=objective, timeout=3600)
print(f"status={status}")
