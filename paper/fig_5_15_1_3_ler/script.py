import json
import numpy as np
import stim
from pathlib import Path
from syqma import QECSimulator
from tqdm import tqdm

folder = Path(__file__).resolve().parent

span_error_ordering_1q = ["Z", "X", "Y"]
span_error_ordering_2q = ["IZ", "ZI", "ZZ", "IX", "IY", "ZX", "ZY", "XI", "XZ", "YI", "YZ", "XX", "XY", "YX", "YY"]

stabs = [stim.PauliString(pauli) 
       for pauli in 
       ['XXXXXXXXIIIIIII',
        'IXXIXXIIXXIXXII',
        'IIXXIXXIIXXXIXI',
        'IIIIXXXXIIIXXXX',
        'ZZZZIIIIIIIIIII',
        'IZZIZZIIIIIIIII',
        'IIZZIZZIIIIIIII',
        'IIIIZZZZIIIIIII',
        'IZIIZIIIZIIIZII',
        'IIZIIZIIIZIZIII',
        'IIZZIIIIIZZIIII',
        'IIIIZZIIIIIZZII',
        'IIIIIZZIIIIZIZI',
        'IIIIIIZZIIIIIZZ']
]

all_stabiliser_operators_z = []
all_stabiliser_operators_x = []

for s in stabs:
    indices = s.pauli_indices()
    if "Z" in s.__str__():
        pauli_string = "Z" * len(indices)
        all_stabiliser_operators_z.append((pauli_string, indices))
    else:
        pauli_string = "X" * len(indices)
        all_stabiliser_operators_x.append((pauli_string, indices))

logical_qubits_z = [0, 1, 8]
logical_qubits_x = [0, 1, 2, 3, 8, 9, 10]

sim = QECSimulator(n=16)

# |0> logical state preparation
sim.h(2)
sim.pauli_error_1(2)
sim.h(6)
sim.pauli_error_1(6)
sim.h(7)
sim.pauli_error_1(7)
sim.h(14)
sim.pauli_error_1(14)

sim.cnot(6, 9)
sim.pauli_error_2((6, 9))
sim.cnot(7, 11)
sim.pauli_error_2((7, 11))
sim.cnot(14, 3)
sim.pauli_error_2((14, 3))
sim.cnot(3, 1)
sim.pauli_error_2((3, 1))
sim.cnot(2, 11)
sim.pauli_error_2((2, 11))
sim.cnot(7, 12)
sim.pauli_error_2((7, 12))
sim.cnot(2, 0)
sim.pauli_error_2((2, 0))
sim.cnot(9, 1)
sim.pauli_error_2((9, 1))
sim.cnot(11, 6)
sim.pauli_error_2((11, 6))
sim.cnot(14, 7)
sim.pauli_error_2((14, 7))
sim.cnot(0, 8)
sim.pauli_error_2((0, 8))
sim.cnot(9, 2)
sim.pauli_error_2((9, 2))
sim.cnot(7, 13)
sim.pauli_error_2((7, 13))
sim.cnot(6, 14)
sim.pauli_error_2((6, 14))
sim.cnot(11, 7)
sim.pauli_error_2((11, 7))
sim.cnot(13, 5)
sim.pauli_error_2((13, 5))
sim.cnot(7, 10)
sim.pauli_error_2((7, 10))
sim.cnot(11, 4)
sim.pauli_error_2((11, 4))
sim.cnot(1, 13)
sim.pauli_error_2((1, 13))
sim.cnot(6, 7)
sim.pauli_error_2((6, 7))
sim.cnot(1, 8)
sim.pauli_error_2((1, 8))
sim.cnot(3, 11)
sim.pauli_error_2((3, 11))

sim.cnot(1, 15)
sim.pauli_error_2((1, 15))
sim.cnot(2, 15)
sim.pauli_error_2((2, 15))
sim.cnot(10, 15)
sim.pauli_error_2((10, 15))

sim.measure(15)

# |+> logical state preparation
sim_x = QECSimulator(n=20)

sim_x.h(2)
sim_x.pauli_error_1(2)
sim_x.h(4)
sim_x.pauli_error_1(4)
sim_x.h(6)
sim_x.pauli_error_1(6)
sim_x.h(10)
sim_x.pauli_error_1(10)
sim_x.h(13)
sim_x.pauli_error_1(13)
sim_x.h(15)
sim_x.pauli_error_1(15)

sim_x.cnot(4, 1)
sim_x.pauli_error_2((4, 1))
sim_x.cnot(6, 5)
sim_x.pauli_error_2((6, 5))
sim_x.cnot(10, 3)
sim_x.pauli_error_2((10, 3))
sim_x.cnot(13, 11)
sim_x.pauli_error_2((13, 11))
sim_x.cnot(2, 1)
sim_x.pauli_error_2((2, 1))
sim_x.cnot(3, 5)
sim_x.pauli_error_2((3, 5))
sim_x.cnot(4, 7)
sim_x.pauli_error_2((4, 7))
sim_x.cnot(11, 12)
sim_x.pauli_error_2((11, 12))
sim_x.cnot(6, 13)
sim_x.pauli_error_2((6, 13))
sim_x.cnot(3, 0)
sim_x.pauli_error_2((3, 0))
sim_x.cnot(5, 4)
sim_x.pauli_error_2((5, 4))
sim_x.cnot(11, 9)
sim_x.pauli_error_2((11, 9))
sim_x.cnot(12, 8)
sim_x.pauli_error_2((12, 8))
sim_x.cnot(7, 14)
sim_x.pauli_error_2((7, 14))
sim_x.cnot(6, 7)
sim_x.pauli_error_2((6, 7))
sim_x.cnot(1, 8)
sim_x.pauli_error_2((1, 8))
sim_x.cnot(2, 9)
sim_x.pauli_error_2((2, 9))
sim_x.cnot(5, 11)
sim_x.pauli_error_2((5, 11))
sim_x.cnot(4, 12)
sim_x.pauli_error_2((4, 12))
sim_x.cnot(13, 14)
sim_x.pauli_error_2((13, 14))
sim_x.cnot(1, 0)
sim_x.pauli_error_2((1, 0))
sim_x.cnot(2, 3)
sim_x.pauli_error_2((2, 3))
sim_x.cnot(9, 10)
sim_x.pauli_error_2((9, 10))
sim_x.cnot(15, 1)
sim_x.pauli_error_2((15, 1))
sim_x.cnot(15, 3)
sim_x.pauli_error_2((15, 3))
sim_x.cnot(15, 5)
sim_x.pauli_error_2((15, 5))
sim_x.cnot(15, 7)
sim_x.pauli_error_2((15, 7))
sim_x.cnot(15, 9)
sim_x.pauli_error_2((15, 9))
sim_x.cnot(15, 12)
sim_x.pauli_error_2((15, 12))
sim_x.cnot(15, 13)
sim_x.pauli_error_2((15, 13))

sim_x.h(15)
sim_x.pauli_error_1(15)

sim_x.h(17)
sim_x.pauli_error_1(17)

sim_x.cnot(0, 16)
sim_x.pauli_error_2((0, 16))
sim_x.cnot(17, 16)
sim_x.pauli_error_2((17, 16))
sim_x.cnot(5, 16)
sim_x.pauli_error_2((5, 16))
sim_x.cnot(9, 16)
sim_x.pauli_error_2((9, 16))
sim_x.cnot(17, 16)
sim_x.pauli_error_2((17, 16))
sim_x.cnot(14, 16)
sim_x.pauli_error_2((14, 16))

sim_x.h(17)
sim_x.pauli_error_1(17)

sim_x.h(19)
sim_x.pauli_error_1(19)

sim_x.cnot(3, 18)
sim_x.pauli_error_2((3, 18))
sim_x.cnot(19, 18)
sim_x.pauli_error_2((19, 18))
sim_x.cnot(7, 18)
sim_x.pauli_error_2((7, 18))
sim_x.cnot(9, 18)
sim_x.pauli_error_2((9, 18))
sim_x.cnot(19, 18)
sim_x.pauli_error_2((19, 18))
sim_x.cnot(12, 18)
sim_x.pauli_error_2((12, 18))

sim_x.h(19)
sim_x.pauli_error_1(19)

sim_x.measure(15)
sim_x.measure(16)
sim_x.measure(17)
sim_x.measure(18)
sim_x.measure(19)

stabiliser_operators_z = all_stabiliser_operators_z
stabiliser_operators_x = all_stabiliser_operators_x

print(f"len(stabiliser_operators_z) = {len(stabiliser_operators_z)}")
print(f"len(stabiliser_operators_x) = {len(stabiliser_operators_x)}")

print("Doing measurements...")

sim.measure_stabilisers_virtually(stabiliser_operators_z)
sim_x.measure_stabilisers_virtually(stabiliser_operators_x)

# measure in Z basis
output_z = sim.measure_all(logical_qubits_z)

# measure in X basis
for qubit in logical_qubits_x:
    sim_x.h(qubit)
output_x = sim_x.measure_all(logical_qubits_x)
for qubit in logical_qubits_x:
    sim_x.h(qubit)

all_acceptance_probabilities_z = []
all_acceptance_probabilities_x = []
all_exp_val_z_free = []
all_exp_val_z_postselected_ancilla = []
all_exp_val_z_postselected_stabs = []
all_exp_val_z_corrected = []
all_exp_val_x_free = []
all_exp_val_x_postselected_ancilla = []
all_exp_val_x_postselected_stabs = []
all_exp_val_x_corrected = []

all_p0 = np.logspace(-3, 0, 51)

print("Calculating noisy EVs...")

for i, p0 in tqdm(enumerate(all_p0), total=len(all_p0)):
    std_dev_scale = 0.0

    p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
    errors_1q = np.hstack((1 - p0, p_base_1q))
    p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
    errors_2q = np.hstack((1 - p0, p_base_2q))

    exp_val_z_free = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_z, measured_pauli_string="Z" * len(logical_qubits_z), postselected_indices=None, output=output_z)
    exp_val_z_postselected_ancilla = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_z, measured_pauli_string="Z" * len(logical_qubits_z), postselected_indices=list(range(sim.n_m - len(stabiliser_operators_z))), output=output_z)
    exp_val_z_postselected_stabs = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_z, measured_pauli_string="Z" * len(logical_qubits_z), postselected_indices=list(range(sim.n_m)), output=output_z)
    
    exp_val_x_free = sim_x.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_x, measured_pauli_string="X" * len(logical_qubits_x), postselected_indices=None, output=output_x)
    exp_val_x_postselected_ancilla = sim_x.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_x, measured_pauli_string="X" * len(logical_qubits_x), postselected_indices=list(range(sim_x.n_m - len(stabiliser_operators_x))), output=output_x)
    exp_val_x_postselected_stabs = sim_x.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits_x, measured_pauli_string="X" * len(logical_qubits_x), postselected_indices=list(range(sim_x.n_m)), output=output_x)
    
    syndrome_probabilities_z = {}
    syndrome_probabilities_x = {}
    lut_z = {}
    lut_x = {}
    corrections = {}
    
    syndrome_probabilities_z = sim.get_all_marginal_probabilities(errors_1q, errors_2q, postselected_indices=list(range(sim.n_m - len(stabiliser_operators_z))))
    syndrome_probabilities_x = sim_x.get_all_marginal_probabilities(errors_1q, errors_2q, postselected_indices=list(range(sim_x.n_m - len(stabiliser_operators_x))))
    
    lut_z = sim.lut_exp_vals_from_syndromes(("Z" * len(logical_qubits_z), logical_qubits_z), list(range(sim.n_m)), errors_1q, errors_2q, output=output_z, postselected_indices=list(range(sim.n_m - len(stabiliser_operators_z))))
    lut_x = sim_x.lut_exp_vals_from_syndromes(("X" * len(logical_qubits_x), logical_qubits_x), list(range(sim_x.n_m)), errors_1q, errors_2q, output=output_x, postselected_indices=list(range(sim_x.n_m - len(stabiliser_operators_x))))
    
    assert np.isclose(sum(list(syndrome_probabilities_z.values())), 1.0)
    assert np.isclose(sum(list(syndrome_probabilities_x.values())), 1.0)
    
    exp_val_z_probabilities = sum([v * lut_z[k] for k, v in syndrome_probabilities_z.items()])
    exp_val_x_probabilities = sum([v * lut_x[k] for k, v in syndrome_probabilities_x.items()])
    
    assert np.isclose(exp_val_z_probabilities, exp_val_z_postselected_ancilla)
    assert np.isclose(exp_val_x_probabilities, exp_val_x_postselected_ancilla)
    
    exp_val_z_corrected = sum([v * abs(lut_z[k]) for k, v in syndrome_probabilities_z.items()])
    exp_val_x_corrected = sum([v * abs(lut_x[k]) for k, v in syndrome_probabilities_x.items()])
    
    all_exp_val_z_free.append(exp_val_z_free)
    all_exp_val_z_postselected_ancilla.append(exp_val_z_postselected_ancilla)
    all_exp_val_z_postselected_stabs.append(exp_val_z_postselected_stabs)
    all_exp_val_z_corrected.append(exp_val_z_corrected)
    
    all_exp_val_x_free.append(exp_val_x_free)
    all_exp_val_x_postselected_ancilla.append(exp_val_x_postselected_ancilla)
    all_exp_val_x_postselected_stabs.append(exp_val_x_postselected_stabs)
    all_exp_val_x_corrected.append(exp_val_x_corrected)
    
    acceptance_probability_z = syndrome_probabilities_z[tuple([0] * len(stabiliser_operators_z))]
    acceptance_probability_x = syndrome_probabilities_x[tuple([0] * len(stabiliser_operators_x))]
    all_acceptance_probabilities_z.append(acceptance_probability_z)
    all_acceptance_probabilities_x.append(acceptance_probability_x)

all_exp_val_z_free = [float(v) for v in all_exp_val_z_free]
all_exp_val_z_postselected_ancilla = [float(v) for v in all_exp_val_z_postselected_ancilla]
all_exp_val_z_postselected_stabs = [float(v) for v in all_exp_val_z_postselected_stabs]
all_exp_val_z_corrected = [float(v) for v in all_exp_val_z_corrected]
all_acceptance_probabilities_z = [float(v) for v in all_acceptance_probabilities_z]
all_acceptance_probabilities_x = [float(v) for v in all_acceptance_probabilities_x]

all_exp_val_x_free = [float(v) for v in all_exp_val_x_free]
all_exp_val_x_postselected_ancilla = [float(v) for v in all_exp_val_x_postselected_ancilla]
all_exp_val_x_postselected_stabs = [float(v) for v in all_exp_val_x_postselected_stabs]
all_exp_val_x_corrected = [float(v) for v in all_exp_val_x_corrected]

all_ler_z_free = (1 - np.array(all_exp_val_z_free)) / 2
all_ler_z_postselected_ancilla = (1 - np.array(all_exp_val_z_postselected_ancilla)) / 2
all_ler_z_postselected_stabs = (1 - np.array(all_exp_val_z_postselected_stabs)) / 2
all_ler_z_corrected = (1 - np.array(all_exp_val_z_corrected)) / 2
all_discard_probabilities_z = 1 - np.array(all_acceptance_probabilities_z)
all_discard_probabilities_x = 1 - np.array(all_acceptance_probabilities_x)

all_ler_x_free = (1 - np.array(all_exp_val_x_free)) / 2
all_ler_x_postselected_ancilla = (1 - np.array(all_exp_val_x_postselected_ancilla)) / 2
all_ler_x_postselected_stabs = (1 - np.array(all_exp_val_x_postselected_stabs)) / 2
all_ler_x_corrected = (1 - np.array(all_exp_val_x_corrected)) / 2

d = {
    "p0": all_p0.tolist(),
    "free_z": all_ler_z_free.tolist(),
    "postselected_ancilla_z": all_ler_z_postselected_ancilla.tolist(),
    "postselected_stabs_z": all_ler_z_postselected_stabs.tolist(),
    "corrected_z": all_ler_z_corrected.tolist(),
    "free_x": all_ler_x_free.tolist(),
    "postselected_ancilla_x": all_ler_x_postselected_ancilla.tolist(),
    "postselected_stabs_x": all_ler_x_postselected_stabs.tolist(),
    "corrected_x": all_ler_x_corrected.tolist(),
    "discard_z": all_discard_probabilities_z.tolist(),
    "discard_x": all_discard_probabilities_x.tolist()
}

with open(folder / "data.json", "w") as f:
    json.dump(d, f, indent=4)
