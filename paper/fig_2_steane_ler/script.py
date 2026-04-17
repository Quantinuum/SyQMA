import json
import numpy as np
from pathlib import Path
from syqma.qec_simulator import QECSimulator

folder = Path(__file__).resolve().parent

# ordering of error terms in the channels
span_error_ordering_1q = ["Z", "X", "Y"]
span_error_ordering_2q = ["IZ", "ZI", "ZZ", "IX", "IY", "ZX", "ZY", "XI", "XZ", "YI", "YZ", "XX", "XY", "YX", "YY"]

# number of physical qubits
n = 8

# define the qubits support for stabilisers and logical operators in the Steane code
steane_stabilisers = [[0, 1, 2, 3], [1, 2, 4, 5], [2, 3, 5, 6]]
steane_logicals = [[0, 1, 4], [0, 3, 6], [4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

# choose one logical representative
logical_qubits = steane_logicals[0]

sim = QECSimulator(n=n)

# |0> logical state preparation
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

# projectively measure the ancilla
sim.measure(7)

# define and measure Z stabilisers virtually (no ancillas)
z_stabilisers = [("Z" * len(qubits), qubits) for qubits in steane_stabilisers]
sim.measure_stabilisers_virtually(z_stabilisers)

# measure Z logical operator in Z basis and store the solutions in the output
output_z = sim.measure_all(logical_qubits)

all_single_probabilities = []
all_acceptance_probabilities = []
all_exp_val_z_free = []
all_exp_val_z_postselected = []
all_exp_val_z_corrected = []

# define the base physical error rates
# all_p0 = np.logspace(-5, np.log10(0.5), 101)
all_p0 = np.logspace(-5, 0, 101)
# all_p0 = [1e-3]

for p0 in all_p0:
    # variance in the error distribution, 0 is depolarising channel
    std_dev_scale = 0.0

    # generate the error probabilities for the 1q and 2q Pauli channels
    p_base_1q = np.random.normal(p0 / 3, std_dev_scale * p0 / 3, 3)
    errors_1q = np.hstack((1 - p0, p_base_1q))
    p_base_2q = np.random.normal(p0 / 15, std_dev_scale * p0 / 15, 15)
    errors_2q = np.hstack((1 - p0, p_base_2q))

    # calculate the expectation values of the logical operators in the case of only postselecting on the ancilla
    exp_val_z_free = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="Z" * len(logical_qubits), postselected_indices=[0], output=output_z)
    
    # calculate the expectation values of the logical operators in the case of postselecting on the ancilla andall the stabilisers
    exp_val_z_postselected = sim.postselected_expectation_value(errors_1q, errors_2q, measured_qubits=logical_qubits, measured_pauli_string="Z" * len(logical_qubits), postselected_indices=list(range(len(z_stabilisers) + 1)), output=output_z)
    
    syndrome_probabilities_z = {}
    lut_z = {}
    
    # get the probabilities of all the syndromes, conditioned on the ancilla being 0
    syndrome_probabilities_z = sim.get_all_marginal_probabilities(errors_1q, errors_2q, postselected_indices=[0])
    
    # get the LUTs of expectation values based on syndromes, also conditioned on the ancilla being 0
    lut_z = sim.lut_exp_vals_from_syndromes(("Z" * len(logical_qubits), logical_qubits), list(range(len(z_stabilisers) + 1)), errors_1q, errors_2q, output=output_z, postselected_indices=[0])
    
    # check that the sum of the probabilities of all the syndromes is 1.0
    assert np.isclose(sum(list(syndrome_probabilities_z.values())), 1.0)
    
    # calculate the expectation values of the logical operators by summing over all the syndrome-dependent expectation values
    exp_val_z_probabilities = sum([v * lut_z[k] for k, v in syndrome_probabilities_z.items()])
    
    assert np.isclose(exp_val_z_probabilities, exp_val_z_free)
    
    # calculate the expectation values of the logical operators after perfect error correction
    exp_val_z_corrected = sum([v * abs(lut_z[k]) for k, v in syndrome_probabilities_z.items()])
    
    all_exp_val_z_free.append(exp_val_z_free)
    all_exp_val_z_postselected.append(exp_val_z_postselected)
    all_exp_val_z_corrected.append(exp_val_z_corrected)
    
    acceptance_probability = syndrome_probabilities_z[tuple([0] * len(z_stabilisers))]
    all_acceptance_probabilities.append(acceptance_probability)

    measurement_results = np.zeros(sim.n_m)
    single_probability = sim.get_marginal_probabilities(errors_1q, errors_2q, measurement_results=measurement_results)
    all_single_probabilities.append(single_probability)

all_exp_val_z_free = [float(v) for v in all_exp_val_z_free]
all_exp_val_z_postselected = [float(v) for v in all_exp_val_z_postselected]
all_exp_val_z_corrected = [float(v) for v in all_exp_val_z_corrected]
all_acceptance_probabilities = [float(v) for v in all_acceptance_probabilities]
all_single_probabilities = [float(v) for v in all_single_probabilities]

all_ler_z_free = (1 - np.array(all_exp_val_z_free)) / 2
all_ler_z_postselected = (1 - np.array(all_exp_val_z_postselected)) / 2
all_ler_z_corrected = (1 - np.array(all_exp_val_z_corrected)) / 2
all_discard_probabilities = 1 - np.array(all_acceptance_probabilities)

d = {
    "p0": all_p0.tolist(),
    "free": all_ler_z_free.tolist(),
    "postselected": all_ler_z_postselected.tolist(),
    "corrected": all_ler_z_corrected.tolist(),
    "discard": all_discard_probabilities.tolist(),
}

with open(folder / "data.json", "w") as f:
    json.dump(d, f, indent=4)
