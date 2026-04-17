import json
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from scipy import stats
from syqma.qec_simulator import QECSimulator

folder = Path(__file__).resolve().parent

n = 20

logical_qubits = list(range(15))

z_stabs_15 = [
    [0, 1, 2, 3],
    [1, 2, 4, 5],
    [2, 3, 5, 6],
    [4, 5, 6, 7],
    [1, 4, 8, 12],
    [2, 5, 9, 11],
    [2, 3, 9, 10],
    [4, 5, 11, 12],
    [5, 6, 11, 13],
    [6, 7, 13, 14]
]

x_stabs_15 = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [1, 2, 4, 5, 8, 9, 11, 12],
    [2, 3, 5, 6, 9, 10, 11, 13],
    [4, 5, 6, 7, 11, 12, 13, 14]
]

sim = QECSimulator(n=n, timing=False)

sim.h(2)
sim.pauli_error_1(2)
sim.h(4)
sim.pauli_error_1(4)
sim.h(6)
sim.pauli_error_1(6)
sim.h(10)
sim.pauli_error_1(10)
sim.h(13)
sim.pauli_error_1(13)

sim.cnot(4, 1)
sim.pauli_error_2((4, 1))
sim.cnot(6, 5)
sim.pauli_error_2((6, 5))
sim.cnot(13, 11)
sim.pauli_error_2((13, 11))

sim.cnot(2, 1)
sim.pauli_error_2((2, 1))
sim.cnot(10, 3)
sim.pauli_error_2((10, 3))
sim.cnot(11, 12)
sim.pauli_error_2((11, 12))

sim.cnot(4, 7)
sim.pauli_error_2((4, 7))
sim.cnot(11, 9)
sim.pauli_error_2((11, 9))

sim.cnot(3, 5)
sim.pauli_error_2((3, 5))
sim.cnot(6, 13)
sim.pauli_error_2((6, 13))

sim.cnot(3, 0)
sim.pauli_error_2((3, 0))
sim.cnot(5, 4)
sim.pauli_error_2((5, 4))
sim.cnot(12, 8)
sim.pauli_error_2((12, 8))

sim.cnot(7, 14)
sim.pauli_error_2((7, 14))

sim.cnot(2, 9)
sim.pauli_error_2((2, 9))
sim.cnot(13, 14)
sim.pauli_error_2((13, 14))

sim.cnot(2, 3)
sim.pauli_error_2((2, 3))
sim.cnot(5, 11)
sim.pauli_error_2((5, 11))

sim.cnot(1, 8)
sim.pauli_error_2((1, 8))
sim.cnot(9, 10)
sim.pauli_error_2((9, 10))

sim.cnot(1, 0)
sim.pauli_error_2((1, 0))
sim.cnot(4, 12)
sim.pauli_error_2((4, 12))

sim.cnot(6, 7)
sim.pauli_error_2((6, 7))

sim.h(15)
sim.pauli_error_1(15)

sim.cnot(15, 1)
sim.pauli_error_2((15, 1))
sim.cnot(15, 3)
sim.pauli_error_2((15, 3))
sim.cnot(15, 5)
sim.pauli_error_2((15, 5))
sim.cnot(15, 7)
sim.pauli_error_2((15, 7))
sim.cnot(15, 9)
sim.pauli_error_2((15, 9))
sim.cnot(15, 12)
sim.pauli_error_2((15, 12))
sim.cnot(15, 13)
sim.pauli_error_2((15, 13))

sim.h(15)
sim.pauli_error_1(15)

sim.h(17)
sim.pauli_error_1(17)

sim.cnot(0, 16)
sim.pauli_error_2((0, 16))
sim.cnot(17, 16)
sim.pauli_error_2((17, 16))
sim.cnot(5, 16)
sim.pauli_error_2((5, 16))
sim.cnot(9, 16)
sim.pauli_error_2((9, 16))
sim.cnot(17, 16)
sim.pauli_error_2((17, 16))
sim.cnot(14, 16)
sim.pauli_error_2((14, 16))

sim.h(17)
sim.pauli_error_1(17)

sim.h(19)
sim.pauli_error_1(19)

sim.cnot(3, 18)
sim.pauli_error_2((3, 18))
sim.cnot(19, 18)
sim.pauli_error_2((19, 18))
sim.cnot(7, 18)
sim.pauli_error_2((7, 18))
sim.cnot(9, 18)
sim.pauli_error_2((9, 18))
sim.cnot(19, 18)
sim.pauli_error_2((19, 18))
sim.cnot(12, 18)
sim.pauli_error_2((12, 18))

sim.h(19)
sim.pauli_error_1(19)

for i in range(15):
    sim.rz(i, 0)
    sim.pauli_error_1(i)

start_time = time.time()
sim.measure(15)
end_time = time.time()
print(f"Time taken to measure 15: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(16)
end_time = time.time()
print(f"Time taken to measure 16: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(17)
end_time = time.time()
print(f"Time taken to measure 17: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(18)
end_time = time.time()
print(f"Time taken to measure 18: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(19)
end_time = time.time()
print(f"Time taken to measure 19: {end_time - start_time} seconds")

x_stabs_15_strings = []
x_stabs_15_strings = [("X" * len(stab), stab) for stab in x_stabs_15]
sim.measure_stabilisers_virtually(x_stabs_15_strings)

z_stabs_15_strings = []
z_stabs_15_strings = [("Z" * len(stab), stab) for stab in z_stabs_15]
sim.measure_stabilisers_virtually(z_stabs_15_strings)

from pympler import asizeof

print(f"\nSize of sim: {asizeof.asizeof(sim) / 10 ** 6} MB\n")

base_angle = np.pi / 4
thetas = [base_angle]

print(f"n_m = {sim.n_m}")
postselection_indices = list(range(sim.n_m))
print(f"postselection_indices = {postselection_indices}")

all_acceptance_probabilities = []
all_exp_val_z_free = []
all_exp_val_z_postselected = []
all_exp_val_z_corrected = []
all_exp_val_x_free = []
all_exp_val_x_postselected = []
all_exp_val_x_corrected = []
all_exp_val_y_free = []
all_exp_val_y_postselected = []
all_exp_val_y_corrected = []

n_points = 33
all_p0 = np.logspace(-4, 0, n_points)

all_errors_1q = []
all_errors_2q = []

for p0 in all_p0:
    std_dev_scale = 0.0

    p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
    errors_1q = np.hstack((1 - p0, p_base_1q))
    p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
    errors_2q = np.hstack((1 - p0, p_base_2q))

    all_errors_1q.append(errors_1q)
    all_errors_2q.append(errors_2q)

batch_size = 1024
n_jobs = 1

one_error_start_time = time.time()

start_time = time.time()

for qubit in logical_qubits:
    sim.h(qubit)

lut_x_dicts, probs_dicts = sim.lut_exp_vals_from_syndromes_memoryless(all_errors_1q, all_errors_2q, logical_qubits, postselected_indices=list(range(sim.n_m - len(x_stabs_15_strings) - len(z_stabs_15_strings))), all_thetas=[thetas], batch_size=batch_size, n_jobs=n_jobs)
lut_x = lut_x_dicts

for qubit in logical_qubits:
    sim.h(qubit)

for qubit in logical_qubits:
    sim.sdag(qubit)
    sim.h(qubit)

lut_y_dicts, probs_dicts = sim.lut_exp_vals_from_syndromes_memoryless(all_errors_1q, all_errors_2q, logical_qubits, postselected_indices=list(range(sim.n_m - len(x_stabs_15_strings) - len(z_stabs_15_strings))), all_thetas=[thetas], batch_size=batch_size, n_jobs=n_jobs)
lut_y = lut_y_dicts

for qubit in logical_qubits:
    sim.h(qubit)
    sim.s(qubit)

end_time = time.time()
print(f"Time taken to calculate LUTs: {end_time - start_time} seconds")

print(f"\nSize of lut_x: {asizeof.asizeof(lut_x) / 10 ** 6} MB\n")

all_syndrome_probabilities = probs_dicts

for i_error in range(len(all_errors_1q)):
    rescaled_prob = sum(all_syndrome_probabilities[i_error].values())
    syndrome_probabilities = {k: v / rescaled_prob for k, v in all_syndrome_probabilities[i_error].items()}

    exp_val_x_free = sum([v * lut_x[i_error][k] for k, v in syndrome_probabilities.items()])
    exp_val_x_postselected = lut_x[i_error][tuple([0] * len(list(lut_x[i_error].keys())[0]))]
    exp_val_x_corrected = np.sign(exp_val_x_postselected) * sum([v * abs(lut_x[i_error][k]) for k, v in syndrome_probabilities.items()])
    exp_val_x_ideal = np.sign(exp_val_x_postselected) / np.sqrt(2)

    exp_val_y_free = sum([v * lut_y[i_error][k] for k, v in syndrome_probabilities.items()])
    exp_val_y_postselected = lut_y[i_error][tuple([0] * len(list(lut_y[i_error].keys())[0]))]
    exp_val_y_corrected = np.sign(exp_val_y_postselected) * sum([v * abs(lut_y[i_error][k]) for k, v in syndrome_probabilities.items()])
    exp_val_y_ideal = np.sign(exp_val_y_postselected) / np.sqrt(2)

    all_exp_val_x_free.append(exp_val_x_free)
    all_exp_val_x_postselected.append(exp_val_x_postselected)
    all_exp_val_x_corrected.append(exp_val_x_corrected)
    all_exp_val_y_free.append(exp_val_y_free)
    all_exp_val_y_postselected.append(exp_val_y_postselected)
    all_exp_val_y_corrected.append(exp_val_y_corrected)

    acceptance_probability = syndrome_probabilities[tuple([0] * (len(x_stabs_15_strings) + len(z_stabs_15_strings)))]
    all_acceptance_probabilities.append(acceptance_probability)

one_error_end_time = time.time()
print(f"Total time taken for one error rate: {one_error_end_time - one_error_start_time} seconds")

all_exp_val_x_free = [float(v) for v in all_exp_val_x_free]
all_exp_val_x_postselected = [float(v) for v in all_exp_val_x_postselected]
all_exp_val_x_corrected = [float(v) for v in all_exp_val_x_corrected]
all_exp_val_y_free = [float(v) for v in all_exp_val_y_free]
all_exp_val_y_postselected = [float(v) for v in all_exp_val_y_postselected]
all_exp_val_y_corrected = [float(v) for v in all_exp_val_y_corrected]

exp_vals = {
    "all_p0": all_p0.tolist(),
    "exp_val_x_free": all_exp_val_x_free,
    "exp_val_x_postselected": all_exp_val_x_postselected,
    "exp_val_x_corrected": all_exp_val_x_corrected,
    "exp_val_y_free": all_exp_val_y_free,
    "exp_val_y_postselected": all_exp_val_y_postselected,
    "exp_val_y_corrected": all_exp_val_y_corrected,
    "acceptance_probabilities": all_acceptance_probabilities
}

with open(folder / "data.json", "w") as f:
    json.dump(exp_vals, f, indent=4)

all_p0 = np.array(exp_vals["all_p0"])
all_exp_val_x_free = np.array(exp_vals["exp_val_x_free"])
all_exp_val_x_postselected = np.array(exp_vals["exp_val_x_postselected"])
all_exp_val_x_corrected = np.array(exp_vals["exp_val_x_corrected"])
all_exp_val_y_free = np.array(exp_vals["exp_val_y_free"])
all_exp_val_y_postselected = np.array(exp_vals["exp_val_y_postselected"])
all_exp_val_y_corrected = np.array(exp_vals["exp_val_y_corrected"])
all_acceptance_probabilities = np.array(exp_vals["acceptance_probabilities"])

all_exp_val_y_free = np.abs(all_exp_val_y_free)
all_exp_val_y_postselected = np.abs(all_exp_val_y_postselected)
all_exp_val_y_corrected = np.abs(all_exp_val_y_corrected)

noiseless_exp_val = np.cos(base_angle)

all_ler_free = (1 - (all_exp_val_x_free + all_exp_val_y_free) * noiseless_exp_val) / 2
all_ler_postselected = (1 - (all_exp_val_x_postselected + all_exp_val_y_postselected) * noiseless_exp_val) / 2
all_ler_corrected = (1 - (all_exp_val_x_corrected + all_exp_val_y_corrected) * noiseless_exp_val) / 2

all_discard_probabilities = 1 - np.array(all_acceptance_probabilities)

n_fitted = 3

fit_free = stats.linregress(np.log(all_p0)[:n_fitted], np.log(all_ler_free)[:n_fitted])
fit_postselected = stats.linregress(np.log(all_p0)[:n_fitted], np.log(all_ler_postselected)[:n_fitted])
fit_corrected = stats.linregress(np.log(all_p0)[:n_fitted], np.log(all_ler_corrected)[:n_fitted])

fit_discard = stats.linregress(np.log(all_p0)[:n_fitted], np.log(all_discard_probabilities)[:n_fitted])

# default is 10
plt.rcParams.update({'font.size': 15})

colors = plt.get_cmap("tab10")
markers = ["o", "^", "s", "D", "X", "v"]
markersize = 10
# default is 2
linewidth = 3

fig = plt.figure(figsize=(16, 8))

plt.plot(all_p0, all_discard_probabilities, linewidth=linewidth, marker=markers[3], markersize=markersize, markeredgecolor="black", color=colors(0), label=rf"Discard probability = $p^{{{fit_discard.slope:.2f}}}$")
plt.plot(all_p0, all_ler_free, linewidth=linewidth, marker=markers[0], markersize=markersize, markeredgecolor="black", color=colors(1), label=rf"$X, Y$ free $\to p^{{{fit_free.slope:.2f}}}$")
plt.plot(all_p0, all_ler_corrected, linewidth=linewidth, marker=markers[2], markersize=markersize, markeredgecolor="black", color=colors(2), label=rf"$X, Y$ corrected $\to p^{{{fit_corrected.slope:.2f}}}$")
plt.plot(all_p0, all_ler_postselected, linewidth=linewidth, marker=markers[1], markersize=markersize, markeredgecolor="black", color=colors(3), label=rf"$X, Y$ postselected $\to p^{{{fit_postselected.slope:.2f}}}$")

plt.xlabel("Physical error rate")
plt.ylabel("Logical error rate")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

plt.show()