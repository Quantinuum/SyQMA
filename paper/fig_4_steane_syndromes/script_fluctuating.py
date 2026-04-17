import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from pathlib import Path
from syqma.qec_simulator import QECSimulator

folder = Path(__file__).resolve().parent

n = 8

steane_stabilisers = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
steane_logicals = [[0, 1, 4], [0, 3, 6], [4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

sim = QECSimulator(n=n)

# state preparation for Steane [7, 1, 3] code from Quantinuum paper
sim.h(0)
sim.pauli_error_1(0)
sim.h(4)
sim.pauli_error_1(4)
sim.h(6)
sim.pauli_error_1(6)
sim.cnot(0, 1)
sim.pauli_error_2((0, 1))
sim.cnot(4, 5)
sim.pauli_error_2((4, 5))
sim.cnot(6, 3)
sim.pauli_error_2((6, 3))
sim.cnot(4, 2)
sim.pauli_error_2((4, 2))
sim.cnot(6, 5)
sim.pauli_error_2((6, 5))
sim.cnot(0, 3)
sim.pauli_error_2((0, 3))
sim.cnot(4, 1)
sim.pauli_error_2((4, 1))
sim.cnot(3, 2)
sim.pauli_error_2((3, 2))

sim.cnot(1, 7)
sim.pauli_error_2((1, 7))
sim.cnot(3, 7)
sim.pauli_error_2((3, 7))
sim.cnot(5, 7)
sim.pauli_error_2((5, 7))

sim.measure(7)

logical_qubits = steane_logicals[0]

stabilisers_z = [("Z" * len(qubits), qubits) for qubits in steane_stabilisers]
stabilisers_x = [("X" * len(qubits), qubits) for qubits in steane_stabilisers]

stabilisers = stabilisers_z + stabilisers_x
sim.measure_stabilisers_virtually(stabilisers)

all_prob_0 = []
all_exp_val_z_free = []
all_exp_val_z_postselected = []
all_syndrome_probabilities_z = {k: [] for k in itertools.product([0, 1], repeat=len(stabilisers))}
all_lut_z = {k: [] for k in itertools.product([0, 1], repeat=len(stabilisers))}

all_p0 = np.logspace(np.log10(1e-3), np.log10(1), 101)

normalise = False
normalise_by_trace = True

for p0 in tqdm.tqdm(all_p0):
    std_dev_scale = 0.1

    p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
    errors_1q = np.hstack((1 - p0, p_base_1q))
    p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
    errors_2q = np.hstack((1 - p0, p_base_2q))

    syndrome_probabilities_z = sim.get_all_marginal_probabilities(errors_1q, errors_2q, postselected_indices=list(range(sim.n_m - len(stabilisers))), printing=False)
    lut_z = sim.lut_exp_vals_from_syndromes(("Z" * len(logical_qubits), logical_qubits), list(range(sim.n_m)), errors_1q, errors_2q, postselected_indices=list(range(sim.n_m - len(stabilisers))), printing=False)
    
    for k, v in syndrome_probabilities_z.items():
        all_syndrome_probabilities_z[k].append(float(v))

    for k, v in lut_z.items():
        all_lut_z[k].append(float(v))

print()
for k, v in all_lut_z.items():
    print(f"{k}: ev = {' ' if v[0] > 0 else ''}{v[0]:.15f} | prob = {all_syndrome_probabilities_z[k][0]:.15f}")

bitstrings = []
probs = []
evs = []

for k, v in all_syndrome_probabilities_z.items():
    probs.append(v)

for k, v in all_lut_z.items():
    bitstrings.append(k)
    evs.append(v)

probs = np.array(probs)
bitstrings = np.array(bitstrings)
evs = np.array(evs)

ev = np.zeros(len(all_p0))

for i_s in range(len(probs)):
    for i_p in range(len(all_p0)):
        ev[i_p] += probs[i_s][i_p] * abs(evs[i_s][i_p])

# sort by first ev
bitstrings = bitstrings[np.argsort(evs, axis=0)[:, 0]]
probs = probs[np.argsort(evs, axis=0)[:, 0]]
evs = np.sort(evs, axis=0)

d = {
    "all_p0": all_p0.tolist(),
    "bitstrings": bitstrings.tolist(),
    "probs": probs.tolist(),
    "evs": evs.tolist(),
}

with open(folder / "data_fluctuating.json", "w") as f:
    json.dump(d, f, indent=4)
