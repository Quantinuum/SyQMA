import json
import numpy as np
from pathlib import Path
from syqma import QECSimulator

folder = Path(__file__).resolve().parent

p0 = 1e-2
std_dev_scale = 0.0

p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
errors_1q = np.hstack((1 - p0, p_base_1q))
p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
errors_2q = np.hstack((1 - p0, p_base_2q))


n_datas = [16, 32, 64, 128, 256, 512, 1024]
discard_rates_original_prep = []
discard_rates_half = []
discard_rates_quarter = []
discard_rates_original_syndrome = []
discard_rates_parallel_syndrome = []
discard_rates_prep_syndrome = []
exp_val_logical_z = []
exp_val_logical_z_postselected = []

for n_data in n_datas:
    # original state preparation circuit
    n_ancilla = 1
    n = n_data + n_ancilla

    sim = QECSimulator(n=n)

    sim.h(0)
    sim.pauli_error_1(0)
    for i in range(n_data - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    sim.cnot(0, n_data)
    sim.pauli_error_2((0, n_data))
    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    
    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_original_prep.append(discard_rate)
    
    # half-depth state preparation circuit
    n_ancilla = 1
    n = n_data + n_ancilla

    sim = QECSimulator(n=n)

    sim.h(0)
    sim.pauli_error_1(0)
    sim.cnot(0, n_data // 2)
    sim.pauli_error_2((0, n_data // 2))
    for i in range(n_data // 2 - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for i in range(n_data // 2, n_data - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    sim.cnot(n_data // 2 - 1, n_data)
    sim.pauli_error_2((n_data // 2 - 1, n_data))

    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_half.append(discard_rate)
    
    # quarter-depth state preparation circuit
    n_ancilla = 2
    n = n_data + n_ancilla

    sim = QECSimulator(n=n)

    sim.h(0)
    sim.pauli_error_1(0)
    sim.cnot(0, n_data // 2)
    sim.pauli_error_2((0, n_data // 2))
    sim.cnot(0, n_data // 4)
    sim.pauli_error_2((0, n_data // 4))
    sim.cnot(n_data // 2, n_data // 2 + (2 + n_data) // 4)
    sim.pauli_error_2((n_data // 2, n_data // 2 + (2 + n_data) // 4))

    for i in range(n_data // 4 - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for i in range(n_data // 4, n_data // 2 - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for i in range(n_data // 2, n_data // 2 + (2 + n_data) // 4 - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for i in range(n_data // 2 + (2 + n_data) // 4, n_data - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))

    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    sim.cnot(n_data // 2 + (2 + n_data) // 4 - 1, n_data + 1)
    sim.pauli_error_2((n_data // 2 + (2 + n_data) // 4 - 1, n_data + 1))
    sim.cnot(n_data // 2 - 1, n_data)
    sim.pauli_error_2((n_data // 2 - 1, n_data))
    sim.cnot(n_data // 4 - 1, n_data + 1)
    sim.pauli_error_2((n_data // 4 - 1, n_data + 1))
    
    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_quarter.append(discard_rate)
    
    # original syndrome measurement circuit
    n_ancilla = 2
    n = n_data + n_ancilla
    n_gates = 2 * n

    sim = QECSimulator(n=n)
    
    sim.h(n_data + 1)
    sim.pauli_error_1(n_data + 1)
    sim.cnot(n_data + 1, 0)
    sim.pauli_error_2((n_data + 1, 0))
    sim.cnot(0, n_data)
    sim.pauli_error_2((0, n_data))
    sim.cnot(1, n_data)
    sim.pauli_error_2((1, n_data))
    sim.cnot(n_data + 1, 1)
    sim.pauli_error_2((n_data + 1, 1))
    for i in range(2, n_data - 2):
        sim.cnot(n_data + 1, i)
        sim.pauli_error_2((n_data + 1, i))
        sim.cnot(i, n_data)
        sim.pauli_error_2((i, n_data))
    sim.cnot(n_data + 1, n_data - 2)
    sim.pauli_error_2((n_data + 1, n_data - 2))
    sim.cnot(n_data - 2, n_data)
    sim.pauli_error_2((n_data - 2, n_data))
    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    sim.cnot(n_data + 1, n_data - 1)
    sim.pauli_error_2((n_data + 1, n_data - 1))
    sim.h(n_data + 1)
    sim.pauli_error_1(n_data + 1)
    
    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_original_syndrome.append(discard_rate)
    
    # parallel ZX syndrome measurement circuit
    n_ancilla = 2
    n = n_data + n_ancilla

    sim = QECSimulator(n=n)
    
    sim.h(n_data + 1)
    
    all_pairs = []
    for i in range(0, n_data - 1, 2):  # roughly a staircase
        all_pairs += [(i, n_data), (n_data + 1, i + 1), (i + 1, n_data), (n_data + 1, i)]
    for i in range(-5, -len(all_pairs) // 2, -4):  # swap the gates as indicated in the slide
        all_pairs[i * 2], all_pairs[i * 2 + 2] = all_pairs[i * 2 + 2], all_pairs[i * 2]
        all_pairs[i * 2 + 1], all_pairs[i * 2 + 3] = all_pairs[i * 2 + 3], all_pairs[i * 2 + 1]

    for pair in all_pairs:
        sim.cnot(pair[0], pair[1])
    
    sim.h(n_data + 1)
    
    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_parallel_syndrome.append(discard_rate)
    
    ##########################################################################################

    n_ancilla = 2
    n = n_data + 2 * n_ancilla
    
    sim = QECSimulator(n=n)

    sim.h(0)
    sim.pauli_error_1(0)
    for i in range(n_data - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    sim.cnot(0, n_data)
    sim.pauli_error_2((0, n_data))
    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    
    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    sim.h(n_data + 1)
    sim.pauli_error_1(n_data + 1)
    sim.cnot(n_data + 1, 0)
    sim.pauli_error_2((n_data + 1, 0))
    sim.cnot(0, n_data)
    sim.pauli_error_2((0, n_data))
    sim.cnot(1, n_data)
    sim.pauli_error_2((1, n_data))
    sim.cnot(n_data + 1, 1)
    sim.pauli_error_2((n_data + 1, 1))
    for i in range(2, n_data - 2):
        sim.cnot(n_data + 1, i)
        sim.pauli_error_2((n_data + 1, i))
        sim.cnot(i, n_data)
        sim.pauli_error_2((i, n_data))
    sim.cnot(n_data + 1, n_data - 2)
    sim.pauli_error_2((n_data + 1, n_data - 2))
    sim.cnot(n_data - 2, n_data)
    sim.pauli_error_2((n_data - 2, n_data))
    sim.cnot(n_data - 1, n_data)
    sim.pauli_error_2((n_data - 1, n_data))
    sim.cnot(n_data + 1, n_data - 1)
    sim.pauli_error_2((n_data + 1, n_data - 1))
    sim.h(n_data + 1)
    sim.pauli_error_1(n_data + 1)

    for i in range(n_ancilla):
        sim.measure(i + n_data)
    
    acceptance_rate = sim.acceptance_probability(errors_1q, errors_2q, postselected_indices=list(range(2 * n_ancilla)))
    discard_rate = 1 - acceptance_rate
    discard_rates_prep_syndrome.append(discard_rate)

discard_rates_original_prep = [float(v) for v in discard_rates_original_prep]
discard_rates_half = [float(v) for v in discard_rates_half]
discard_rates_quarter = [float(v) for v in discard_rates_quarter]
discard_rates_original_syndrome = [float(v) for v in discard_rates_original_syndrome]
discard_rates_prep_syndrome = [float(v) for v in discard_rates_prep_syndrome]

d = {
    "n_datas": n_datas,
    "original_prep": discard_rates_original_prep,
    "half_prep": discard_rates_half,
    "quarter_prep": discard_rates_quarter,
    "original_syndrome": discard_rates_original_syndrome,
    "prep_syndrome": discard_rates_prep_syndrome
}

with open(folder / "data.json", "w") as f:
    json.dump(d, f, indent=4)
