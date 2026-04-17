"""This is an older version of the code that has not been checked recently."""

import tracemalloc
import numpy as np
import time
from syqma.qec_simulator import QECSimulator

n = 20

logical_qubits = list(range(15))

z_stabs_span = [
    [0, 1, 2, 3],
    [1, 2, 4, 5],
    [2, 3, 9, 10],
    [2, 3, 5, 6],
    [4, 5, 6, 7],
    [1, 4, 8, 12],
    [2, 5, 9, 11],
    [5, 6, 11, 13],
    [4, 5, 11, 12],
    [6, 7, 13, 14]
]
x_stabs_span = [
    [ 0,  1,  2,  3,  8,  9, 10],
    [ 0,  1,  4,  7,  8, 12, 14],
    [ 4,  5,  6,  7, 11, 12, 13, 14],
    [ 0,  3,  4,  5, 10, 11, 12],
    [ 8,  9, 10, 11, 12, 13, 14]
]

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

x_stabs_15_strings = [("X" * len(stab), stab) for stab in x_stabs_15]
z_stabs_15_strings = [("Z" * len(stab), stab) for stab in z_stabs_15]

stabs_strings = z_stabs_15_strings + x_stabs_15_strings

n_stabs_max = 0
batch_sizes = [1, 2 ** (15 + n_stabs_max)]

all_memory_max = np.zeros(n_stabs_max + 1)
all_time_max = np.zeros(n_stabs_max + 1)

all_memory_max_exp_val = np.zeros(n_stabs_max + 1)
all_time_max_exp_val = np.zeros(n_stabs_max + 1)

all_memory_1 = np.zeros(n_stabs_max + 1)
all_time_1 = np.zeros(n_stabs_max + 1)

all_memory_batch = np.zeros(n_stabs_max + 1)
all_time_batch = np.zeros(n_stabs_max + 1)

filename = "data.txt"

p0 = 1e-3

normalise = False
normalise_by_trace = True

base_angle = np.pi / 4
thetas = [base_angle]

std_dev_scale = 0.0

p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
errors_1q = np.hstack((1 - p0, p_base_1q))
p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
errors_2q = np.hstack((1 - p0, p_base_2q))

print(f"For 1q: min = {np.min(p_base_1q)}, max = {np.max(p_base_1q)}, mean = {np.mean(p_base_1q)}, 1 - p = {errors_1q[0]}")
print(f"For 2q: min = {np.min(p_base_2q)}, max = {np.max(p_base_2q)}, mean = {np.mean(p_base_2q)}, 1 - p = {errors_2q[0]}")

n_stabs = 0

if all_memory_max[n_stabs] == 0:
    tracemalloc.start()

    start_time_sim = time.time()

    start_time = time.time()

    sim = QECSimulator(n=n)

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

    sim.measure_stabilisers_virtually(stabs_strings[:n_stabs])

    from pympler import asizeof

    print(f"\nSize of sim: {asizeof.asizeof(sim) / 10 ** 6} MB\n")

    print("Measuring trace...")

    start_time = time.time()
    output_trace = sim.measure_all([])
    end_time = time.time()
    print(f"Time taken to measure trace: {end_time - start_time} seconds")

    sim.output_trace = output_trace

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

    postselection_indices = list(range(sim.n_m))

    print("\nCalculating free and postselected expectation values...\n")

    current, peak_measure = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()

    start_time_exp_val = time.time()

    exp_val_x_postselected = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="X" * len(logical_qubits), postselected_indices=postselection_indices, thetas=thetas, output=output_x)
    
    end_time = time.time()

    current, peak_exp_val = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    all_memory_max[n_stabs] = max(peak_measure, peak_exp_val) / 10 ** 6
    all_time_max[n_stabs] = end_time - start_time_sim
    all_memory_max_exp_val[n_stabs] = peak_exp_val / 10 ** 6
    all_time_max_exp_val[n_stabs] = end_time - start_time_exp_val
else:
    print(f"Skipping batch_size=max n_stabs = {n_stabs} because it has already been calculated")

if all_memory_1[n_stabs] == 0:
    for i_batch, batch_size in enumerate(batch_sizes):
        tracemalloc.stop()
        tracemalloc.start()

        start_time = time.time()

        sim = QECSimulator(n=n)

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

        sim.measure_stabilisers_virtually(stabs_strings[:n_stabs])

        for q in logical_qubits:
            sim.h(q)
        
        postselection_indices = list(range(sim.n_m))

        print(f"postselection_indices = {postselection_indices}")

        exp_val_x_postselected, prob_0 = sim.evaluate_expectation_values_fixed_memoryless(logical_qubits, [errors_1q], [errors_2q], all_measurement_results=[np.zeros(sim.n_m).tolist()], postselected_indices=postselection_indices, all_thetas=[thetas], batch_size=batch_size)

        for q in logical_qubits:
            sim.h(q)

        end_time = time.time()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
else:
    print(f"Skipping batch_size=1 n_stabs = {n_stabs} because it has already been calculated")

for n_stabs in range(n_stabs_max + 1):
    if all_memory_max[n_stabs] == 0:
        tracemalloc.start()

        start_time = time.time()

        sim = QECSimulator(n=n)

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

        sim.measure_stabilisers_virtually(stabs_strings[:n_stabs])

        from pympler import asizeof

        print(f"\nSize of sim: {asizeof.asizeof(sim) / 10 ** 6} MB\n")

        print("Measuring trace...")

        start_time = time.time()
        output_trace = sim.measure_all([])
        end_time = time.time()
        print(f"Time taken to measure trace: {end_time - start_time} seconds")

        sim.output_trace = output_trace

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

        postselection_indices = list(range(sim.n_m))

        print("\nCalculating free and postselected expectation values...\n")

        current, peak_measure = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()

        start_time_exp_val = time.time()

        exp_val_x_postselected = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="X" * len(logical_qubits), postselected_indices=postselection_indices, thetas=thetas, output=output_x)
        
        end_time = time.time()

        current, peak_exp_val = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_time = time.time()

        total_time = end_time - start_time
        total_time_exp_val = end_time - start_time_exp_val

        print(f"Memory batch_size=max = {max(peak_measure, peak_exp_val) / 10 ** 6} MB")
        print(f"Memory exp val batch_size=max = {peak_exp_val / 10 ** 6} MB")

        print(f"Time taken to calculate free and postselected expectation values: {total_time} seconds\n")
        print(f"Time taken to calculate free and postselected expectation values only exl_val: {total_time_exp_val} seconds")

        print(f"\nX postselected = {exp_val_x_postselected}\n")

        all_memory_max[n_stabs] = max(peak_measure, peak_exp_val) / 10 ** 6
        all_time_max[n_stabs] = total_time
        all_memory_max_exp_val[n_stabs] = peak_exp_val / 10 ** 6
        all_time_max_exp_val[n_stabs] = total_time_exp_val
    else:
        print(f"Skipping batch_size=max n_stabs = {n_stabs} because it has already been calculated")
    
    if all_memory_1[n_stabs] == 0:
        for i_batch, batch_size in enumerate(batch_sizes):
            tracemalloc.stop()
            tracemalloc.start()

            start_time = time.time()

            sim = QECSimulator(n=n)

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

            sim.measure_stabilisers_virtually(stabs_strings[:n_stabs])

            for q in logical_qubits:
                sim.h(q)

            postselection_indices = list(range(sim.n_m))

            exp_val_x_postselected, prob_0 = sim.evaluate_expectation_values_from_scratch(logical_qubits, [errors_1q], [errors_2q], all_measurement_results=[np.zeros(sim.n_m).tolist()], postselected_indices=postselection_indices, all_thetas=[thetas], batch_size=batch_size)

            for q in logical_qubits:
                sim.h(q)

            end_time = time.time()

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"Memory batch_size={batch_size} = {peak / 10 ** 6} MB")

            total_time = end_time - start_time

            print(f"Time taken to calculate free and postselected expectation values memoryless (batch_size={batch_size}): {total_time} seconds")

            print(f"\nX postselected = {exp_val_x_postselected}\n")
            print(f"\nProbability of all 0 = {prob_0}\n")

            all_memory_batch[n_stabs] = peak / 10 ** 6
            all_time_batch[n_stabs] = total_time

            if batch_size == 1:
                all_memory_1[n_stabs] = peak / 10 ** 6
                all_time_1[n_stabs] = total_time
    else:
        print(f"Skipping batch_size=1 n_stabs = {n_stabs} because it has already been calculated")

np.savetxt(filename, np.array([all_memory_max, all_time_max, all_memory_max_exp_val, all_time_max_exp_val, all_memory_1, all_time_1, all_memory_batch, all_time_batch]))
