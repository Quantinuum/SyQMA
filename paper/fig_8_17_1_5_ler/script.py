import json
import numpy as np
import os
import time
from pytket.circuit import Circuit
from pathlib import Path
from syqma.utils import circuit_from_pytket

folder = Path(__file__).resolve().parent

start_all_time = time.time()

z_stabs_17 = [
    [0, 1, 2, 3],
    [1, 3, 6, 7],
    [4, 8, 12, 13],
    [4, 5, 8, 9],
    [2, 3, 5, 6, 9, 10, 14, 15],
    [6, 7, 10, 11],
    [8, 9, 13, 14],
    [10, 11, 15, 16]
]

x_stabs_17 = [
    [0, 1, 2, 3],
    [1, 3, 6, 7],
    [4, 8, 12, 13],
    [4, 5, 8, 9],
    [2, 3, 5, 6, 9, 10, 14, 15],
    [6, 7, 10, 11],
    [8, 9, 13, 14],
    [10, 11, 15, 16]
]

z_stabs_17 = [[x + 21 for x in stab] for stab in z_stabs_17]
x_stabs_17 = [[x + 21 for x in stab] for stab in x_stabs_17]

logical_qubits = [21 + x for x in [12, 13, 14, 15, 16]]

with open(os.path.dirname(__file__) + "/SeventeenColorCode.json", 'r') as f:
    circuit_json = json.load(f)
circuit = Circuit.from_dict(circuit_json)
circuit.flatten_registers()

n = circuit.n_qubits

start_time = time.time()
sim = circuit_from_pytket(circuit, noise=True)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

stabilisers_z = [("Z" * len(stab), stab) for stab in z_stabs_17]
stabilisers_x = [("X" * len(stab), stab) for stab in x_stabs_17]

stabilisers = stabilisers_z

sim.measure_stabilisers_virtually(stabilisers)

print(f"n_m = {sim.n_m}")
print(sim.measurement_qubits)

from pympler import asizeof

print(f"\nSize of sim: {asizeof.asizeof(sim) / 10 ** 6} MB\n")

postselection_indices = list(range(21))
print(f"postselection_indices = {postselection_indices}")

all_acceptance_probabilities = []
all_exp_val_z_free = []
all_exp_val_z_postselected = []
all_exp_val_z_corrected = []

n_points = 17

all_p0 = np.logspace(-4, 0, n_points)

normalise = False
normalise_by_trace = True

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
n_jobs = 10

all_lut_z, all_syndrome_probabilities = sim.lut_exp_vals_from_syndromes_memoryless(all_errors_1q, all_errors_2q, measured_qubits=logical_qubits, postselected_indices=postselection_indices, batch_size=batch_size, n_jobs=n_jobs)

all_exp_val_z_free = []
all_exp_val_z_corrected = []
all_exp_val_z_postselected = []
all_acceptance_probabilities = []

for i in range(len(all_errors_1q)):
    rescaled_prob = sum(all_syndrome_probabilities[i].values())
    syndrome_probabilities = {k: v / rescaled_prob for k, v in all_syndrome_probabilities[i].items()}
    exp_val_z_postselected = all_lut_z[i][(0,) * len(stabilisers)]
    exp_val_z_free = sum([v * all_lut_z[i][k] for k, v in syndrome_probabilities.items()])
    exp_val_z_corrected = np.sign(exp_val_z_postselected) * sum([v * abs(all_lut_z[i][k]) for k, v in syndrome_probabilities.items()])
    acceptance_probability = syndrome_probabilities[(0,) * len(stabilisers)]
    all_exp_val_z_free.append(exp_val_z_free)
    all_exp_val_z_corrected.append(exp_val_z_corrected)
    all_exp_val_z_postselected.append(exp_val_z_postselected)
    all_acceptance_probabilities.append(acceptance_probability)

end_all_time = time.time()
print(f"\n\nAbsolute total time taken: {end_all_time - start_all_time} seconds\n\n")

all_exp_val_z_free = [float(v) for v in all_exp_val_z_free]
all_exp_val_z_postselected = [float(v) for v in all_exp_val_z_postselected]
all_exp_val_z_corrected = [float(v) for v in all_exp_val_z_corrected]

all_acceptance_probabilities = [float(v) for v in all_acceptance_probabilities]

exp_vals = {
    "all_p0": all_p0.tolist(),
    "exp_val_z_free": all_exp_val_z_free,
    "exp_val_z_postselected": all_exp_val_z_postselected,
    "exp_val_z_corrected": all_exp_val_z_corrected,
    "acceptance_probabilities": all_acceptance_probabilities
}

with open(folder / "data.json", "w") as f:
    json.dump(exp_vals, f, indent=4)
