import json
import pandas as pd

data = []
missing = 0
timeouts = 0
with open("/home/yizhak/Research/Code/gt4vnn/verify/results/answers_log.txt") as fr:
    for line in fr:
        if "missing" in line: 
            missing += 1
            continue
        data.append(line.strip())
print(f"missing: {missing}")
# missing: 213, timeouts: 2

parsed_data = []
for d in data:
    # print(d)
    answer = json.loads(d)
    sample, epsilon = answer["sample"], answer["epsilon"]
    res_p, time_p, cex_p = answer["res_p"], answer["time_p"], answer["cex_p"]
    res_f, time_f, cex_f = answer["res_f"], answer["time_f"], answer["cex_f"]
    parsed_data.append(
        [sample, epsilon, res_f, res_p, time_f, time_p, cex_f, cex_p]
    )

columns = ["sample", "epsilon", "res_f", "res_p", "time_f", "time_p", "cex_f", "cex_p"]
df = pd.DataFrame(columns=columns, data=parsed_data)
df.time_f = df.time_f.apply(lambda x: float(x))
df.time_p = df.time_p.apply(lambda x: float(x))
print(f"total: {df.shape[0]}")
df_notnull = df.dropna(subset={'res_p', 'res_f'})
print(f"total not null: {df_notnull.shape[0]}")

# correctness
equal = df[df.res_f == df.res_p]
equal_sat = equal[equal.res_f=="sat"]
equal_unsat = equal[equal.res_f=="unsat"]
print(f"correct: {equal.shape[0]}")
# correct: 371
print(f"correct sat: {equal_sat.shape[0]}")
# correct sat: 123
print(f"correct unsat: {equal_unsat.shape[0]}")
# correct unsat: 248
print(f"correct percent: {equal.shape[0]/df_notnull.shape[0]}")
# correct percent: 0.9946380697050938
# correct percent: 0.9946380697050938
print(f"incorrect: {df_notnull[df_notnull.res_f!=df_notnull.res_p].shape[0]}")
# incorrect: 2
print(f"p=sat, f=unsat: {df[(df.res_f=='unsat') & (df.res_p=='sat')].shape[0]}")
# incorrect sat: 2
print(f"p=unsat, f=sat: {df[(df.res_f=='sat') & (df.res_p=='unsat')].shape[0]}")
# incorrect unsat: 0
print(f"incorrect percent: {df_notnull[df_notnull.res_f!=df_notnull.res_p].shape[0]/df_notnull.shape[0]}")
# incorrect percent: 0.005361930294906166
print(f"p=sat, f=timeout: {df[(df.res_p=='sat') & (df.res_f.isin([None]))].shape[0]}")
print(f"p=unsat, f=timeout: {df[(df.res_p=='unsat') & (df.res_f.isin([None]))].shape[0]}")

# time
print(f"percent of better time: {df_notnull[df_notnull.time_f > df_notnull.time_p].shape[0] / df_notnull.shape[0]*100}%")
# percent of better time: 100.0%
print(f"full avg time: {equal.time_f.mean()}, partial avg time: {equal.time_p.mean()}")
# full avg time: 1.9897560855977976, partial avg time: 0.3495489023326229
print(f"sat time percent: {(equal_sat.time_p/equal_sat.time_f).mean()}")
# sat time percent: 0.08807285131789312
print(f"unsat time percent: {(equal_unsat.time_p/equal_unsat.time_f).mean()}")
# unsat time percent: 0.6834178959628621


# check satisfying examples
import torch
import numpy as np
from data_loaders import get_testloader
from verify.verify_utils import get_trained_network
from config import input_size, hidden_sizes, BATCH_SIZE

project_dir = "/home/yizhak/Research/Code/gt4vnn"
suffix = "_".join([str(hs) for hs in hidden_sizes])
torch_path = f"{project_dir}/mnist_fc_sc_clf_net_SC-CL_{suffix}.pth"
PARTIAL = True
tnet, onet = get_trained_network(torch_path, validate=True, partial=PARTIAL)

testloader = get_testloader()
preds = {}
gts = {}
for j, (input_data, labels) in enumerate(testloader):
    input_data = torch.Tensor(input_data.view(-1,input_size))
    preds[j] = tnet.forward(input_data).argmax(axis=1)
    gts[j] = labels
    
def check_cex_p(row):
    cex_p = {int(k): v for k,v in row.cex_p.items()}
    sample = row["sample"]
    orig_pred = preds[sample // BATCH_SIZE][sample % BATCH_SIZE]
    x = np.array(list(cex_p.values()))
    y = tnet.forward(torch.Tensor(x)).argmax()
    return y != orig_pred

df_sat = df[df.res_p=="sat"]
df_sat["is_real_sat"] = df_sat.apply(check_cex_p, axis=1)
print(f"real partial sats: {df_sat.is_real_sat.value_counts}")
print(f"real partial percent: {df_sat[df_sat.is_real_sat==True].shape[0]/df_sat.shape[0]}")
