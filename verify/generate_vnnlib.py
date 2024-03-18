import numpy as np
from typing import Sequence
from config import input_size, output_size


def generate_robustness_property(x: np.array, y:np.array, epsilon :float = 0.001) -> str:
    """generates robustness properties for input x and its corresponding output y.
    Args:
        X (np.ndarray): the input.
        y (np.ndarray): the output
        epsilon (float, optional): the epsilon of the robustness property
    Returns:
        string containing the vnnlib property for the given data.
    """
    assert input_size == len(x)
    prop_s = "; Verifying network robustness for a single input\n\n"
    # prop_s += "; " + str(x) + "\n" + str(y) + "\n" + str(epsilon) + "\n\n"
    # input vars declaration
    for i in range(input_size):
        prop_s += f"(declare-const X_{i} Real)\n"
    prop_s+="\n"
    # output vars declaration
    for i in range(output_size):
        prop_s += f"(declare-const Y_{i} Real)\n"
    prop_s+="\n"
    # input constraint
    epsilon = np.ones(input_size) * epsilon
    for i in range(input_size):
        prop_s += f"(assert (<= X_{i} {x[i]+epsilon[i]}))\n"
        prop_s += f"(assert (>= X_{i} {x[i]-epsilon[i]}))\n"
    # output constraints
    prop_s += "\n(assert (or\n"
    for i in range(output_size):
        if i != y:
            prop_s += f"(and (>= Y_{i} Y_{y}))\n"
    prop_s += "))"
    return prop_s
