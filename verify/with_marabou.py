# ASSUME:
# property is in vnnlib format
# network is in torch format
# Marabou is the verifier
import pandas as pd
import json
import time
import onnx
import onnxruntime
import torch
import tempfile
import subprocess
from maraboupy import Marabou, MarabouCore, MarabouUtils
import numpy as np
from pathlib import Path
import sys, os
from verify.read_vnnlib import (
    Property, get_num_inputs_outputs, read_vnnlib_simple
)
from data_loaders import get_testloader
from verify.verify_utils import get_trained_network
from verify.generate_vnnlib import generate_robustness_property
from config import PROPERTY_FORMAT, input_size, hidden_sizes


# epsilon results are for the first sample in the dataset
e1 = 0.35  # SAT
e2 = 0.0005  # SAT
e3 = 0.00005  # UNSAT
# e4 = 0.000005 # UNSAT in preprocess
epsilons = [e1, e2, e3]  #, e4]


def read_onnx_by_marabou(onnx_model: onnx.ModelProto):
    """load marabou network from onnx"""
    tmp_onnx_file = tempfile.NamedTemporaryFile(mode='wb')
    onnx.save(onnx_model, tmp_onnx_file.name)
    network = Marabou.read_onnx(tmp_onnx_file.name)
    tmp_onnx_file.close()
    return network


def add_robustness_input_region(network, x, epsilon):
    """add input constraints"""
    # Set the input bounds to be an delta ball (in l1-norm ) around a data point
    input_vars = network.inputVars[0].flatten()
    assert len(input_vars) == len(x)
    for i in range(len(x)):
        network.setLowerBound(input_vars[i], x[i] - epsilon)
        network.setUpperBound(input_vars[i], x[i] + epsilon)


def add_classification_output_condition(network, y, delta):
    """add doutput constraints (single runner or disjunction)"""
    # y as the winner index
    output_vars = network.outputVars[0][0]
    winner_var = output_vars[y]
    runners_vars = output_vars[:y].tolist() + output_vars[y+1:].tolist()
    disjunction = []
    for var in runners_vars:
        # eq = MarabouCore.Equation(MarabouCore.Equation.GE)  # older version
        eq = MarabouUtils.Equation(MarabouCore.Equation.GE)  # latest version
        eq.addAddend(1, var)
        eq.addAddend(-1, winner_var)
        eq.setScalar(delta)
        disjunction.append([eq])
    network.addDisjunctionConstraint(disjunction)


def evaluate_with_onnx(onnx_net_path, x):
    import onnxruntime as ort
    ORT_session = ort.InferenceSession(onnx_net_path)
    def run_model(input_data):
        input_data = input_data.astype(np.float32).reshape(1,-1)
        input_name = ORT_session.get_inputs()[0].name
        output_name = ORT_session.get_outputs()[0].name
        result = ORT_session.run([output_name], {input_name: input_data})
        return result[0]
    y = run_model(x).argmax()
    # print(f"onnx: y={y}")
    return y


def get_marabou_dir():
    return [s for s in sys.path if "marabou" in s.lower()][-1]


def run_verifier(query_path):
    marabou_dir = get_marabou_dir()
    cmd = f"cd {marabou_dir}/build;./Marabou --input-query {query_path}"
    start = time.time()
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    return res.stderr, res.stdout, time.time()-start


def parse_marabou_output(output):
    # print(output)
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "sat":
            # parse_counter_example
            cex = {  # i+2 since next line = "Input assignment:\n"
                j: float(lines[i+2+j].strip().split(" ")[-1]) 
                for j in range(input_size)
            }
            return "sat", cex
        elif line.strip() == "unsat":
            return "unsat"
    raise Exception("output is neither sat nor unsat")


def answer_all_queries(queries_dir, results_dir):
    # partial_2layers_query_0_epsilon_5e-05.ipq
    results_stderr = {}
    results_stdout = {}
    from itertools import product
    queries = product(range(10000),epsilons)
    layers = len(hidden_sizes)
    with open(f"{results_dir}/answers_log.txt", "w") as fw:
        for sample, eps in queries:
            # if sample < 1216: continue
            print(f"sample={sample}, epsilon={eps}")
            query_f = f"full_{layers}layers_query_{sample}_epsilon_{eps}.ipq"
            query_p = f"partial_{layers}layers_query_{sample}_epsilon_{eps}.ipq"
            query_paths = [f"{queries_dir}/{q}" for q in [query_f, query_p]]
            # compare only if both queries exist
            if any (not os.path.exists(qp) for qp in query_paths):
                print("missing query")
                fw.write(f"sample={sample}, epsilon={eps}, missing query\n")
                continue
            res_f = None
            res_p = None
            time_f = None
            time_p = None
            cex_f = None
            cex_p = None
            try:
                # verify partial
                err_p, out_p, time_p = run_verifier(f"{queries_dir}/{query_p}")
                assert not err_p
                res_p = parse_marabou_output(out_p)
                if len(res_p) == 2:
                    res_p, cex_p = res_p[0], res_p[1]
                # verify full
                err_f, out_f, time_f = run_verifier(f"{queries_dir}/{query_f}")
                assert not err_f
                res_f = parse_marabou_output(out_f)
                if len(res_f) == 2:
                    res_f, cex_f = res_f[0], res_f[1]
            except subprocess.TimeoutExpired:
                print("TimeoutExpired")
            except AssertionError:
                print("AssertionError")
                assertion_counter += 1
            print(f"res_f={res_f}, res_p={res_p}, time_f={time_f}, time_p={time_p}")
            fw.write(json.dumps({
                "sample":sample, "epsilon":eps,
                "res_p": res_p, "time_p":time_p, "cex_p":cex_p,
                "res_f": res_f, "time_f":time_f, "cex_f":cex_f
            })+"\n")
            fw.flush()
    print(f"#assertions: {assertion_counter}")
    # for query in os.listdir(queries_dir):
    #     print(f"query={query}")
    #     err, out = run_verifier(f"{queries_dir}/{query}")
    #     results_stderr[query] = err
    #     results_stdout[query] = parse_marabou_output(out)
    #     assert not err
    #     print(f"out={results_stdout[query]}")
    # results_path = f"{results_dir}/{query.split(".")[0]}"
    # pd.DataFrame(
    #     data=results_stdout.items(), columns={"query", "result"}
    # ).to_json(results_path)
    # return results_path, results_stderr, results_stdout


queries_dir = "/home/yizhak/Research/Code/gt4vnn/verify/queries"
properties_dir = "/home/yizhak/Research/Code/gt4vnn/verify/properties"
results_dir = "/home/yizhak/Research/Code/gt4vnn/verify/results"
for directory in [queries_dir, properties_dir, results_dir]:
    os.makedirs(directory, exist_ok=True)

RUN_VERIFIER_ON_QUERIES = True
if RUN_VERIFIER_ON_QUERIES:
# run all prepared queries
    answer_all_queries(queries_dir, results_dir)
    sys.exit(0)

# else: prepare queries
project_dir = "/home/yizhak/Research/Code/gt4vnn"
suffix = "_".join([str(hs) for hs in hidden_sizes])
torch_path = f"{project_dir}/mnist_fc_sc_clf_net_SC-CL_{suffix}.pth"
PARTIAL = True
tnet, onet = get_trained_network(torch_path, validate=True, partial=PARTIAL)

print(f"PROPERTY_FORMAT = {PROPERTY_FORMAT}")
prefix = "partial" if PARTIAL else "full"
if PROPERTY_FORMAT == "vnnlib":  # save one network and multiple properties
    onnx_dir = "/home/yizhak/Research/Code/gt4vnn/verify/onnx_networks"
    os.makedirs(onnx_dir, exist_ok=True)
    onnx.save(onet, f"{onnx_dir}/{prefix}_net_{len(hidden_sizes)}layers.onnx")

testloader = get_testloader()
for j, (input_data, label) in enumerate(testloader):
    for i in range(len(input_data)):
        index = j*len(input_data) + i
        print(f"index={index}, j={j}, i={i}")
        for epsilon in epsilons:
            x = input_data.view(-1, input_size)[i]
            y = tnet.forward(torch.Tensor(x)).argmax()
            # y = evaluate_with_onnx("/tmp/partial_net.onnx", x)
            # print(f"y={y}")

            # onnx.save(onet, f"{onnx_dir}/{prefix}_net.onnx")
            if PROPERTY_FORMAT == "vnnlib":
                # save network and property
                with open(f"{properties_dir}/adversarial_property_{index}_epsilon={epsilon}.vnnlib", "w") as fw:
                    fw.write(generate_robustness_property(x, y, epsilon=epsilon))
                # ./Marabou /tmp/net.onnx /tmp/adversarial_property.vnnlib
            elif PROPERTY_FORMAT == "adversarial":
                # save query
                mnet = read_onnx_by_marabou(onet)
                add_robustness_input_region(mnet, x, epsilon=epsilon)
                DELTA = 0.0001
                add_classification_output_condition(mnet, y, delta=DELTA)
                # ipq = mnet.getMarabouQuery()
                ipq = mnet.getInputQuery()
                output_path = f"{queries_dir}/{prefix}_{len(hidden_sizes)}layers_query_{index}_epsilon_{epsilon}.ipq"
                MarabouCore.saveQuery(ipq, str(output_path))
                # ./Marabou --input-query /tmp/query.ipq
            else:
                raise Exception(f"invalid elif PROPERTY_FORMAT: {PROPERTY_FORMAT}")
        break  # generate query to the first sample of any batch

# load (onnx) network and (vnnlib) property
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
# benchmarks_dir = Path("/home/yizhak/Research/Code/vnncomp2022_benchmarks/benchmarks/")
# onnx_path = benchmarks_dir / "mnist_fc/onnx/mnist-net_256x2.onnx"
# vnnlib_prop_path = benchmarks_dir / "mnist_fc/vnnlib/prop_9_0.05.vnnlib"
# net, prop = mnistfc_query_from_vnnlibpath(vnnlib_prop_path, onnx_path)
# net = onnx.load(onnx_path)
# mnet = read_onnx_by_marabou(onet)
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
# # TODO: implement add_classification_output_condition() function
# add_classification_output_condition(mnet, property)
# ipq = mnet.getMarabouQuery()
# output_path = "/tmp/query.ipq"
# MarabouCore.saveQuery(ipq, str(output_path))
# # ./Marabou --input-query /tmp/query.ipq
