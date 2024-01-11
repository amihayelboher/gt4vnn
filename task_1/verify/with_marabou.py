# ASSUME:
# property is in vnnlib format
# network is in torch format
# Marabou is the verifier
import onnx
import torch
import tempfile
from maraboupy import Marabou, MarabouCore
from task_1.verify.read_vnnlib import (
    Property, get_num_inputs_outputs, read_vnnlib_simple
)
import numpy as np
from pathlib import Path


benchmarks_dir = Path("/home/yizhak/Research/Code/vnncomp2022_benchmarks/benchmarks/")
# onnx_path = benchmarks_dir / "mnist_fc/onnx/mnist-net_256x2.onnx"
# vnnlib_prop_path = benchmarks_dir / "mnist_fc/vnnlib/prop_9_0.05.vnnlib"
torch_path = "/home/yizhak/Research/Code/gradual_training_for_verification/mnist_net_general_128_64.pth"
from task_1.task_1_utils import FullyConnected
tnet = FullyConnected(784,10,[128,64])
tnet.load_state_dict(torch.load(torch_path))



# def mnistfc_query_from_vnnlibpath(prop_path, onnx_path):
#     """use the code of stanley bak to read vnnlib files"""
#     num_inputs, num_outputs, _ = get_num_inputs_outputs(onnx_path)
#     vnnlib = read_vnnlib_simple(prop_path, num_inputs, num_outputs)
#     lower_bounds = np.array(vnnlib[0][0])[:, 0]
#     upper_bounds = np.array(vnnlib[0][0])[:, 1]
#     winner = vnnlib[0][1][0][0].argmax()
#     network = load_mnistfc_network(onnx_path)
#     # network = onnx.load(network)
#     if network[-1].weight.data.shape[0] == 1:
#         raise Exception("can't define adversarial property for network with one output")
#         # mat (in mat * y <= rhs) is a vector with single 1
#     net_with_1_output = convert_to_one_output(network, winner)
#     prop = Property(
#         input_constraints=(lower_bounds, upper_bounds), 
#         output_constraints=(np.array(0), None)
#     )
#     return net_with_1_output, prop

# after training, save torch model
# load torch model, convert to onnx + reshape, save on tempfile
import torch.nn as nn
def convert_torch_to_onnx(torch_model: nn.Module, input_shape) -> onnx.ModelProto:
    tmp_onnx_file = tempfile.NamedTemporaryFile(mode='wb')
    dummy_input = torch.randn(input_shape, dtype=torch.float32)
    torch.onnx.export(torch_model, dummy_input, tmp_onnx_file.name, verbose=False)
    onnx_model = onnx.load(tmp_onnx_file.name)
    tmp_onnx_file.close()
    return onnx_model
def reshape_onnx_input_batch_to_one(onnx_model: onnx.ModelProto, new_batch_size=1) -> onnx.ModelProto:
    """transform an onnx_model with input shape of [a, b, c]
    to a model with input shape of [1, b, c].
    returns onnx.ModelProto model with one batch in the input
    """
    assert new_batch_size > 0
    input_shape = [x.dim_value for x in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    input_shape[0] = new_batch_size
    output_shape = [x.dim_value for x in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    output_shape[0] = new_batch_size
    input_ = onnx.helper.make_tensor_value_info(onnx_model.graph.input[0].name, onnx.TensorProto.FLOAT, list(input_shape))
    output_ = onnx.helper.make_tensor_value_info(onnx_model.graph.output[0].name, onnx.TensorProto.FLOAT, list(output_shape))
    onnx_model.graph.ClearField('input')
    onnx_model.graph.input.extend([input_])
    onnx_model.graph.ClearField('output')
    onnx_model.graph.output.extend([output_])
    onnx.checker.check_model(onnx_model)
    return onnx_model
onet = convert_torch_to_onnx(tnet, (1, 784))
onet = reshape_onnx_input_batch_to_one(onet)
# load (onnx) network and (vnnlib) property
# net, prop = mnistfc_query_from_vnnlibpath(vnnlib_prop_path, onnx_path)
# net = onnx.load(onnx_path)
# load marabou network from onnx
def read_onnx_by_marabou(onnx_model: onnx.ModelProto):
    tmp_onnx_file = tempfile.NamedTemporaryFile(mode='wb')
    onnx.save(onnx_model, tmp_onnx_file.name)
    network = Marabou.read_onnx(tmp_onnx_file.name)
    tmp_onnx_file.close()
    return network
mnet = read_onnx_by_marabou(onet)
# # add input constraints
# def add_robustness_input_region(network, property):
#     # Set the input bounds to be an epsilon ball (in l1-norm ) around a data point
#     input_vars = network.inputVars[0].flatten()
#     input_constraints = property[0][0]
#     assert len(input_vars) == len(input_constraints)
#     for i in range(len(input_constraints)):
#         network.setLowerBound(input_vars[i], input_constraints[i][0])
#         network.setUpperBound(input_vars[i], input_constraints[i][1])
# add_robustness_input_region(net, prop)

# add input constraints
def add_robustness_input_region(network, x, delta):
    # Set the input bounds to be an delta ball (in l1-norm ) around a data point
    input_vars = network.inputVars[0].flatten()
    assert len(input_vars) == len(x)
    for i in range(len(x)):
        network.setLowerBound(input_vars[i], x[i] - delta)
        network.setUpperBound(input_vars[i], x[i] + delta)
x = np.array(list([0.1]*784))
DELTA = 0.0001
add_robustness_input_region(mnet, x, delta=DELTA)

# add doutput constraints (single runner or disjunction)
def add_classification_output_condition(network, y, epsilon):
    # y as the winner index
    output_vars = network.outputVars[0][0]
    winner_var = output_vars[y]
    runners_vars = output_vars[:y].tolist() + output_vars[y+1:].tolist()
    disjunction = []
    for var in runners_vars:
        eq = MarabouCore.Equation(MarabouCore.Equation.GE)
        eq.addAddend(1, var)
        eq.addAddend(-1, winner_var)
        eq.setScalar(epsilon)
        disjunction.append([eq])
    network.addDisjunctionConstraint(disjunction)
y = tnet.forward(torch.Tensor(x)).argmax()
# import onnxruntime as ort
# ORT_session = ort.InferenceSession(onnx_path)
# def run_model(input_data):
#     input_data = input_data.astype(np.float32)
#     input_name = ORT_session.get_inputs()[0].name
#     output_name = ORT_session.get_outputs()[0].name
#     result = ORT_session.run([output_name], {input_name: input_data})
#     return result[0]
# y = run_model(x).argmax()
print(y)
EPSILON = 0.0001
add_classification_output_condition(mnet, y, epsilon=EPSILON)

ipq = mnet.getMarabouQuery()
output_path = "/tmp/query.ipq"
MarabouCore.saveQuery(ipq, str(output_path))
