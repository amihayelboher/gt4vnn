import onnx
import onnxruntime
import torch
import tempfile
import numpy as np
from pathlib import Path
import torch.nn as nn
from maraboupy import Marabou, MarabouCore
from verify.read_vnnlib import (
    Property, get_num_inputs_outputs, read_vnnlib_simple
)
from data_loaders import get_testloader
from networks import SequentialWithSkipConnection
from networks import FullyConnectedSkipConnection
from config import input_size, output_size, hidden_sizes


benchmarks_dir = Path("/home/yizhak/Research/Code/vnncomp2022_benchmarks/benchmarks/")
# onnx_path = benchmarks_dir / "mnist_fc/onnx/mnist-net_256x2.onnx"
# vnnlib_prop_path = benchmarks_dir / "mnist_fc/vnnlib/prop_9_0.05.vnnlib"

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


def convert_torch_to_onnx(torch_model: nn.Module, input_shape) -> onnx.ModelProto:
    """convert torch network to onnx network"""
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


def get_trained_network(torch_path, validate=True, partial=True):
    """return torch network and onnx network"""
    tnet = FullyConnectedSkipConnection(input_size, output_size, hidden_sizes)
    tnet.load_state_dict(torch.load(torch_path))
    # for partial verification, we trim network's layers
    if partial:
        half = int(len(tnet.layers)/2)
        tnet = SequentialWithSkipConnection(tnet, num_of_layers=half, clf_index=-1)
    onet = convert_torch_to_onnx(tnet, (1, input_size))
    onet = reshape_onnx_input_batch_to_one(onet)
    if validate:
        testloader = get_testloader()
        valid_conversion = check_torch2onnx_conversion(tnet, onet, testloader)
        print(f"valid_conversion = {valid_conversion}")
        assert valid_conversion
    return tnet, onet


def check_torch2onnx_conversion(tnet, onet, testloader):
    # check the correctness of the conversion from torch to onnx
    ort_session = onnxruntime.InferenceSession(onet.SerializeToString())
    for input_data,label in testloader:
        first_input_data = input_data.view(-1, input_size)[0]
        torch_output = tnet(first_input_data)
        first_input_data_np = first_input_data.numpy()
        ort_output = ort_session.run(None, {'onnx::Gemm_0': first_input_data_np.reshape((1,784))})[0]
        # Compare the outputs
        output_match = torch.allclose(
            torch_output.detach(), torch.Tensor(ort_output[0]), 
            rtol=1e-03, atol=1e-05
        )
        if not output_match:
            print("Conversion failed! Outputs differ between PyTorch and ONNX.")
            break
    if output_match:
        print("Conversion successful! Outputs match between PyTorch and ONNX.")
    return output_match


if __name__ == "__main__":
    torch_path = "/home/yizhak/Research/Code/gt4vnn/mnist_fc_sc_clf_net_SC-CL_256_256.pth"
    tnet, onet = get_trained_network(torch_path, validate=True, partial=True)
    # print(f"tnet={tnet}")
    # print(f"onet={onet}")
    