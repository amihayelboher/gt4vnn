import torch
import numpy as np
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

# abs(16-0)-abs(18-16)=14, abs(18-16)-abs(16-13)=-1, abs(16-13)-abs(16-15)=2, abs(16-15)-abs(16-16)=1
total_num_points = 0
ratios = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        total_num_points += len(images)
        outputs = model(images.view(images.size(0), -1))
        # print(f"winners={outputs.argmax(axis=1)}")
        out01_clf1 = model.classifiers[-1](model.intermediate_results[0] + model.intermediate_results[1])
        out0_clf1 = model.classifiers[-1](model.intermediate_results[0])
        # print(f"out01_clf1.argmax(axis=1)={out01_clf1.argmax(axis=1)}")
        # print(f"out0_clf1.argmax(axis=1)={out0_clf1.argmax(axis=1)}")
        for image_index in range(len(images)):
            winner = outputs[image_index].argmax().item()
            target = outputs[image_index][winner].item()
            start = 0  # the starting value of winner before next layer
            clf_input_i = torch.zeros(model.intermediate_results[0][image_index].shape)
            parts = []
            for i,layer in enumerate(model.layers):
                # i'th input to classifier include all previous layers' outputs
                clf_input_i += model.intermediate_results[i][image_index]
                clf_output_i = model.classifiers[-1](clf_input_i)
                # ith part is ammount of advance towards target in layer i
                ith_part = abs(target-start) - abs(target-clf_output_i[winner])
                # print(f"ith_part={ith_part}")
                start = clf_output_i[winner]
                parts.append(ith_part.item())
            # print(f"parts={parts}")
            ratios.append([p/sum(parts) for p in parts])
            # out0_clf1 = model.classifiers[1](model.intermediate_results[0][image_index])
            # l1w = abs(target-start) - abs(target-out0_clf1[winner])
            # print(f"l1w={l1w}")
            # out01_clf1 = model.classifiers[1](model.intermediate_results[0] + model.intermediate_results[1])
            # l2w = abs(target-out0_clf1[winner]) - abs(target-out01_clf1[image_index][winner])
            # print(f"l2w={l2w}")
            # print(f"aaa: {(l1w/(l1w+l2w), l2w/(l1w+l2w))}")
            # ratios.append((l1w/(l1w+l2w), l2w/(l1w+l2w)))
        # break

print(f"total_num_points={total_num_points}")    
print(f"ratios[:10]={ratios[:10]}")

import matplotlib.pyplot as plt

def generate_percentage_graph(ratios):
    from collections import defaultdict
    percentages = [i * 10 for i in range(11)]  # 0%, 10%, 20%, ..., 100%
    layer2counts = np.zeros((len(ratios[0]), len(percentages))) #{l: [0] * len(percentages) for l in range(len(ratios[0]))}

    for ratio in ratios:
        # print(f"ratio={ratio}")
        cumulative_sum = 0
        for l, fraction in enumerate(ratio):
            cumulative_sum += fraction
            for j, percent in enumerate(percentages):
                if cumulative_sum * 100 >= percent:
                    layer2counts[l,j] += 1
    # import pandas as pd
    # df = pd.DataFrame(counts)
    # df.plot(kind="bar")

    # Create the plot
    markers = ["o", "s", "D", "p", "*", "+"]
    colors = ["b", "g", "r", "c", "m", "y"]
    for i, layer in enumerate(model.layers):
        plt.plot(percentages, layer2counts[i]/total_num_points*100, marker=markers[i], color=colors[i], label=f"Layer {i}")
    plt.xlabel('%Total Result')
    plt.ylabel('%Points')
    plt.title('%Points with %Total Result Per Layer')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage with a list of tuples representing fractions
# ratios = [(0.2, 0.3, 0.5), (0.1, 0.2, 0.7), (0.4, 0.4, 0.2)]
generate_percentage_graph(ratios)

# given a list of tuples "ratios", where each couple includes fractions whose sum is 1
# please write code that generates a graph with 
# x-axis that represents percentages (0,10,20,30,...,100) and 
# y-axis that represents ammount of tuples
# such that for each coordinate i the graph shows the ammount of points
# where the sum until the i'th coordinate of the tuple is more than p% of the whole sum of the tuple

# such that for each layer number i \in (0,1) the graph shows ammount of points
# where the ratio of clf(sum_until_layer_i) was more than p% of the output




# out0_clf1 = model.classifiers[1](model.intermediate_results[0])
# out0_clf0 = model.classifiers[0](model.intermediate_results[0])
# out1_clf1 = model.classifiers[1](model.intermediate_results[1])
# out1_clf0 = model.classifiers[0](model.intermediate_results[1])
# out01_clf1 = model.classifiers[1](model.intermediate_results[0] + model.intermediate_results[1])
# out01_clf0 = model.classifiers[0](model.intermediate_results[0] + model.intermediate_results[1])

# print((model.intermediate_results[0] == model.intermediate_results[1]).all())
# # >> False
# print(f"out0_clf0: {[out0_clf0[i].argmax() for i in range(images.shape[0])]}")
# # >> [tensor(7), tensor(2), tensor(1), tensor(0)]
# print(f"out0_clf1: {[out0_clf1[i].argmax() for i in range(images.shape[0])]}")
# # >> out0_clf1: [tensor(7), tensor(2), tensor(8), tensor(0)]
# print(f"out1_clf0: {[out1_clf0[i].argmax() for i in range(images.shape[0])]}")
# # >> out1_clf0: [tensor(8), tensor(4), tensor(4), tensor(6)]
# print(f"out1_clf1: {[out1_clf1[i].argmax() for i in range(images.shape[0])]}")
# # >> out1_clf1: [tensor(8), tensor(2), tensor(4), tensor(4)]
# print(f"out01_clf1: {[out01_clf1[i].argmax() for i in range(images.shape[0])]}")
# # >> out01_clf1: [tensor(7), tensor(2), tensor(8), tensor(0)]
# print(f"out01_clf0: {[out01_clf0[i].argmax() for i in range(images.shape[0])]}")
# # >> out01_clf0: [tensor(7), tensor(2), tensor(1), tensor(0)]
# print(f"outputs: {[outputs[i].argmax() for i in range(images.shape[0])]}")
# # >> [tensor(7), tensor(2), tensor(1), tensor(0)]






# import matplotlib.pyplot as plt

# def generate_plot(data):
#     N = len(data[0])  # Number of percentages in each tuple
#     num_tuples = len(data)

#     # Initialize a 2D array to count occurrences
#     count_matrix = [[0] * N for _ in range(N)]

#     # Count occurrences for each (p, q) pair
#     for percentages in data:
#         cumulative_sum = 0
#         for i, percent in enumerate(percentages):
#             cumulative_sum += percent
#             for j in range(N):
#                 if cumulative_sum >= j + 1:
#                     count_matrix[i][j] += 1
#     print(data)
#     print(count_matrix)

#     # Create the plot
#     fig, ax = plt.subplots()
#     im = ax.imshow(count_matrix, cmap="viridis")

#     # Set axis labels and ticks
#     ax.set_xticks(range(N))
#     ax.set_yticks(range(N))
#     ax.set_xticklabels([f"{i}%" for i in range(1, N + 1)])
#     ax.set_yticklabels([f"{i}" for i in range(N)])

#     # Add colorbar
#     cbar = ax.figure.colorbar(im, ax=ax)
#     cbar.set_label("Number of Tuples", rotation=-90, va="bottom")

#     plt.xlabel("Percentage")
#     plt.ylabel("Threshold")
#     plt.title("Occurrences of Cumulative Percentages")
#     plt.show()

# # Example usage with a list of tuples
# data = [
#     (10, 20, 30, 40),
#     (5, 25, 35, 35),
#     (15, 15, 25, 45),
#     # Add more tuples as needed
# ]

# generate_plot(data)