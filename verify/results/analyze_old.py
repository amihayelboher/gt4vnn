import pandas as pd
data = []
missing = 0
timeouts = 0
with open("/home/yizhak/Research/Code/gt4vnn/verify/results/answers_log_2layers_182-377.txt") as fr:
    for line in fr:
        if line.startswith("sample"): 
            continue
        if line.startswith("missing"): 
            missing += 1
            continue
        if line.startswith("TimeoutExpired"): 
            timeouts += 1
            continue
        data.append(line.strip().split(", "))
print(f"missing: {missing}, timeouts: {timeouts}")
# missing: 213, timeouts: 2

data = [list(c.split(": ")[-1] for c in d) for d in data]
df = pd.DataFrame(columns=["res_f", "res_p", "time_f", "time_p"], data=data)
df.time_f = df.time_f.apply(lambda x: float(x))
df.time_p = df.time_p.apply(lambda x: float(x))
print(f"total: {df.shape[0]}")

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
print(f"correct percent: {equal.shape[0]/df.shape[0]}")
# correct percent: 0.9946380697050938
# correct percent: 0.9946380697050938
print(f"incorrect: {df[df.res_f!=df.res_p].shape[0]}")
# incorrect: 2
print(f"incorrect sat: {df[(df.res_f=='unsat') & (df.res_p=='sat')].shape[0]}")
# incorrect sat: 2
print(f"incorrect unsat: {df[(df.res_f=='sat') & (df.res_p=='unsat')].shape[0]}")
# incorrect unsat: 0
print(f"incorrect percent: {df[df.res_f!=df.res_p].shape[0]/df.shape[0]}")
# incorrect percent: 0.005361930294906166

# time
print(f"percent of better time: {df[df.time_f > df.time_p].shape[0] / df.shape[0]*100}%")
# percent of better time: 100.0%
print(f"full avg time: {df.time_f.mean()}, partial avg time: {df.time_p.mean()}")
# full avg time: 1.9897560855977976, partial avg time: 0.3495489023326229
print(f"sat time percent: {(equal_sat.time_p/equal_sat.time_f).mean()}")
# sat time percent: 0.08807285131789312
print(f"unsat time percent: {(equal_unsat.time_p/equal_unsat.time_f).mean()}")
# unsat time percent: 0.6834178959628621
