import json
import numpy as np
import sys
import time
from pathlib import Path
from syqma.qec_simulator import QECSimulator
from tqdm import tqdm

np.random.seed(0)
np.set_printoptions(linewidth=100000)
np.set_printoptions(threshold=sys.maxsize)

folder = Path(__file__).resolve().parent

start_all_time = time.time()

n = 24

steane_stabilisers = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
steane_logicals = [[0, 1, 4], [0, 3, 6], [4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

logical_qubits = steane_logicals[-1]

sim = QECSimulator(n=n)

# non-FT state preparation for the |H> state
sim.h(0)
sim.pauli_error_1(0)
sim.h(1)
sim.pauli_error_1(1)
sim.ry(2, 0)
sim.pauli_error_1(2)

sim.h(3)
sim.pauli_error_1(3)
sim.cnot(2, 4)
sim.pauli_error_2((2, 4))
sim.cnot(0, 6)
sim.pauli_error_2((0, 6))
sim.cnot(3, 5)
sim.pauli_error_2((3, 5))
sim.cnot(2, 5)
sim.pauli_error_2((2, 5))
sim.cnot(0, 4)
sim.pauli_error_2((0, 4))
sim.cnot(1, 6)
sim.pauli_error_2((1, 6))
sim.cnot(0, 2)
sim.pauli_error_2((0, 2))
sim.cnot(1, 5)
sim.pauli_error_2((1, 5))
sim.cnot(3, 4)
sim.pauli_error_2((3, 4))
sim.cnot(1, 2)
sim.pauli_error_2((1, 2))
sim.cnot(3, 6)
sim.pauli_error_2((3, 6))

# ancillas are 7, 8

sim.h(7)
sim.pauli_error_1(7)

# ch gates have noise applied internally
sim.ch(7, 6, 0)
sim.cnot(7, 8)
sim.pauli_error_2((7, 8))
sim.ch(7, 5, 0)
sim.ch(7, 4, 0)
sim.ch(7, 3, 0)
sim.ch(7, 2, 0)
sim.ch(7, 1, 0)
sim.cnot(7, 8)
sim.pauli_error_2((7, 8))
sim.ch(7, 0, 0)

sim.h(7)
sim.pauli_error_1(7)

p0 = 1e-3
std_dev_scale = 0.0

p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
errors_1q = np.hstack((1 - p0, p_base_1q))
p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
errors_2q = np.hstack((1 - p0, p_base_2q))

base_angle = np.pi / 4
thetas = [base_angle, -base_angle]

start_time = time.time()
sim.measure(7)
end_time = time.time()
print(f"Time taken to measure 7: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(8)
end_time = time.time()
print(f"Time taken to measure 8: {end_time - start_time} seconds")

# ancillas are 9, 10, 11
sim.h(9)
sim.pauli_error_1(9)
sim.cnot(9, 4)
sim.pauli_error_2((9, 4))
sim.cnot(6, 10)
sim.pauli_error_2((6, 10))
sim.cnot(5, 11)
sim.pauli_error_2((5, 11))
sim.cnot(9, 11)
sim.pauli_error_2((9, 11))
sim.cnot(9, 0)
sim.pauli_error_2((9, 0))
sim.cnot(4, 10)
sim.pauli_error_2((4, 10))
sim.cnot(1, 11)
sim.pauli_error_2((1, 11))

sim.cnot(9, 2)
sim.pauli_error_2((9, 2))
sim.cnot(3, 10)
sim.pauli_error_2((3, 10))
sim.cnot(6, 11)
sim.pauli_error_2((6, 11))
sim.cnot(9, 10)
sim.pauli_error_2((9, 10))

sim.cnot(9, 6)
sim.pauli_error_2((9, 6))
sim.cnot(5, 10)
sim.pauli_error_2((5, 10))
sim.cnot(2, 11)
sim.pauli_error_2((2, 11))

sim.h(9)
sim.pauli_error_1(9)

start_time = time.time()
sim.measure(9)
end_time = time.time()
print(f"Time taken to measure 9: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(10)
end_time = time.time()
print(f"Time taken to measure 10: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(11)
end_time = time.time()
print(f"Time taken to measure 11: {end_time - start_time} seconds")

# ancillas are 12, 13, 14
sim.h(13)
sim.pauli_error_1(13)
sim.h(14)
sim.pauli_error_1(14)

sim.cnot(4, 12)
sim.pauli_error_2((4, 12))
sim.cnot(13, 6)
sim.pauli_error_2((13, 6))
sim.cnot(14, 5)
sim.pauli_error_2((14, 5))

sim.cnot(14, 12)
sim.pauli_error_2((14, 12))
sim.cnot(0, 12)
sim.pauli_error_2((0, 12))
sim.cnot(13, 4)
sim.pauli_error_2((13, 4))
sim.cnot(14, 1)
sim.pauli_error_2((14, 1))

sim.cnot(2, 12)
sim.pauli_error_2((2, 12))
sim.cnot(13, 3)
sim.pauli_error_2((13, 3))
sim.cnot(14, 6)
sim.pauli_error_2((14, 6))
sim.cnot(13, 12)
sim.pauli_error_2((13, 12))

sim.cnot(6, 12)
sim.pauli_error_2((6, 12))
sim.cnot(13, 5)
sim.pauli_error_2((13, 5))
sim.cnot(14, 2)
sim.pauli_error_2((14, 2))

sim.h(13)
sim.pauli_error_1(13)
sim.h(14)
sim.pauli_error_1(14)

start_time = time.time()
sim.measure(12)
end_time = time.time()
print(f"Time taken to measure 12: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(13)
end_time = time.time()
print(f"Time taken to measure 13: {end_time - start_time} seconds")

start_time = time.time()
sim.measure(14)
end_time = time.time()
print(f"Time taken to measure 14: {end_time - start_time} seconds")

steane_stabilisers = [[0, 2, 4, 6], [3, 4, 5, 6], [1, 2, 5, 6]]

steane_stabilisers_z_strings = [("Z" * len(stab), stab) for stab in steane_stabilisers]
sim.measure_stabilisers_virtually(steane_stabilisers_z_strings)

steane_stabilisers_x_strings = [("X" * len(stab), stab) for stab in steane_stabilisers]
sim.measure_stabilisers_virtually(steane_stabilisers_x_strings)

from pympler import asizeof

print(f"\nSize of sim: {asizeof.asizeof(sim) / 10 ** 6} MB\n")

print("Measuring logical Z...")

start_time = time.time()
output_z = sim.measure_all(logical_qubits)
end_time = time.time()
print(f"Time taken to measure logical Z: {end_time - start_time} seconds")

print(f"\nSize of output_z: {asizeof.asizeof(output_z) / 10 ** 6} MB\n")

print("Measuring logical X...")

# measure in X basis
start_time = time.time()
for qubit in logical_qubits:
    sim.h(qubit)
output_x = sim.measure_all(logical_qubits)
for qubit in logical_qubits:
    sim.h(qubit)
end_time = time.time()
print(f"Time taken to measure logical X: {end_time - start_time} seconds")

base_angle = np.pi / 4
thetas = [base_angle, -base_angle]

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

all_syndrome_probabilities = []
all_lut_z = []
all_lut_x = []

n_points = 51
all_p0 = np.logspace(-5, 0, n_points)

normalise = False
normalise_by_trace = True

for p0 in tqdm(all_p0, total=len(all_p0), desc="Calculating for p0 = "):
    one_error_start_time = time.time()
    
    std_dev_scale = 0.0

    p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
    errors_1q = np.hstack((1 - p0, p_base_1q))
    p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
    errors_2q = np.hstack((1 - p0, p_base_2q))

    print(f"p0 = {p0}")
    print(f"For 1q: min = {np.min(p_base_1q)}, max = {np.max(p_base_1q)}, mean = {np.mean(p_base_1q)}, 1 - p = {errors_1q[0]}")
    print(f"For 2q: min = {np.min(p_base_2q)}, max = {np.max(p_base_2q)}, mean = {np.mean(p_base_2q)}, 1 - p = {errors_2q[0]}")
    
    exp_val_z_free = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="Z" * len(logical_qubits), postselected_indices=None, thetas=thetas, output=output_z)
    exp_val_z_postselected = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="Z" * len(logical_qubits), postselected_indices=postselection_indices, thetas=thetas, output=output_z)
    
    print(f"Z free = {exp_val_z_free}, Z postselected = {exp_val_z_postselected}")

    exp_val_x_free = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="X" * len(logical_qubits), postselected_indices=None, thetas=thetas, output=output_x)
    exp_val_x_postselected = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="X" * len(logical_qubits), postselected_indices=postselection_indices, thetas=thetas, output=output_x)
    
    print(f"X free = {exp_val_x_free}, X postselected = {exp_val_x_postselected}")
    
    end_time = time.time()
    print(f"Time taken to calculate free and postselected expectation values: {end_time - start_time} seconds")
    
    lut_z = {}
    lut_x = {}
    corrections = {}
    
    print("\nCalculating syndrome probabilities...\n")
    
    start_time = time.time()

    syndrome_probabilities = sim.get_all_marginal_probabilities(errors_1q, errors_2q, thetas, postselected_indices=list(range(sim.n_m - 6)))
    
    end_time = time.time()
    print(f"Time taken to calculate syndrome probabilities: {end_time - start_time} seconds")
    
    print("\nCalculating LUTs...\n")
    
    start_time = time.time()

    lut_z = sim.lut_exp_vals_from_syndromes(("Z" * len(logical_qubits), logical_qubits), list(range(sim.n_m)), errors_1q, errors_2q, thetas=thetas, output=output_z, postselected_indices=list(range(sim.n_m - 6)))
    lut_x = sim.lut_exp_vals_from_syndromes(("X" * len(logical_qubits), logical_qubits), list(range(sim.n_m)), errors_1q, errors_2q, thetas=thetas, output=output_x, postselected_indices=list(range(sim.n_m - 6)))
    
    end_time = time.time()
    print(f"Time taken to calculate LUTs: {end_time - start_time} seconds")

    print(f"\nSize of lut_z: {asizeof.asizeof(lut_z) / 10 ** 6} MB\n")
    print(f"Size of lut_x: {asizeof.asizeof(lut_x) / 10 ** 6} MB\n")

    all_syndrome_probabilities.append(syndrome_probabilities)
    all_lut_z.append(lut_z)
    all_lut_x.append(lut_x)
    
    assert np.isclose(sum(list(syndrome_probabilities.values())), 1.0)

    exp_val_z_probabilities = sum([v * lut_z[k] for k, v in syndrome_probabilities.items()])
    exp_val_x_probabilities = sum([v * lut_x[k] for k, v in syndrome_probabilities.items()])
    
    print(exp_val_z_probabilities, exp_val_z_free)
    print(exp_val_x_probabilities, exp_val_x_free)

    exp_val_z_corrected = sum([v * abs(lut_z[k]) for k, v in syndrome_probabilities.items()])
    exp_val_x_corrected = sum([v * abs(lut_x[k]) for k, v in syndrome_probabilities.items()])

    print(exp_val_z_free, exp_val_z_corrected, exp_val_z_postselected)
    print(exp_val_x_free, exp_val_x_corrected, exp_val_x_postselected)
    
    try:
        assert np.isclose(exp_val_z_probabilities, exp_val_z_free)
        assert np.isclose(exp_val_x_probabilities, exp_val_x_free)
    except Exception as e:
        print(e)

    exp_val_z_free = exp_val_z_probabilities
    exp_val_x_free = exp_val_x_probabilities
    
    all_exp_val_z_free.append(exp_val_z_free)
    all_exp_val_z_postselected.append(exp_val_z_postselected)
    all_exp_val_z_corrected.append(exp_val_z_corrected)
    
    all_exp_val_x_free.append(exp_val_x_free)
    all_exp_val_x_postselected.append(exp_val_x_postselected)
    all_exp_val_x_corrected.append(exp_val_x_corrected)
    
    acceptance_probability = syndrome_probabilities[tuple([0] * 6)]
    all_acceptance_probabilities.append(acceptance_probability)
    
    one_error_end_time = time.time()
    print(f"Total time taken for one error rate: {one_error_end_time - one_error_start_time} seconds")

end_all_time = time.time()
print(f"\n\nAbsolute total time taken: {end_all_time - start_all_time} seconds\n\n")

all_exp_val_z_free = [float(v) for v in all_exp_val_z_free]
all_exp_val_z_postselected = [float(v) for v in all_exp_val_z_postselected]
all_exp_val_z_corrected = [float(v) for v in all_exp_val_z_corrected]

all_exp_val_x_free = [float(v) for v in all_exp_val_x_free]
all_exp_val_x_postselected = [float(v) for v in all_exp_val_x_postselected]
all_exp_val_x_corrected = [float(v) for v in all_exp_val_x_corrected]

all_acceptance_probabilities = [float(v) for v in all_acceptance_probabilities]

exp_vals = {
    "all_p0": all_p0.tolist(),
    "exp_val_z_free": all_exp_val_z_free,
    "exp_val_z_postselected": all_exp_val_z_postselected,
    "exp_val_z_corrected": all_exp_val_z_corrected,
    "exp_val_x_free": all_exp_val_x_free,
    "exp_val_x_postselected": all_exp_val_x_postselected,
    "exp_val_x_corrected": all_exp_val_x_corrected,
    "acceptance_probabilities": all_acceptance_probabilities
}

with open(folder / "data.json", "w") as f:
    json.dump(exp_vals, f, indent=4)