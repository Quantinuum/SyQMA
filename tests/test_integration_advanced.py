import numpy as np
import stim
from syqma import QECSimulator

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

np.random.seed(0)

import sys

np.set_printoptions(linewidth=100000)
np.set_printoptions(threshold=sys.maxsize)

span_error_ordering_1q = ["Z", "X", "Y"]
stim_error_ordering_1q = ["X", "Y", "Z"]
span_error_ordering_2q = ["IZ", "ZI", "ZZ", "IX", "IY", "ZX", "ZY", "XI", "XZ", "YI", "YZ", "XX", "XY", "YX", "YY"]
stim_error_ordering_2q = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]

pauli_strings_1q = ["I", "Z", "X", "Y"]
pauli_strings_2q = ["II", "ZI", "IZ", "ZZ", "XI", "YI", "XZ", "YZ", "IX", "ZX", "IY", "ZY", "XX", "YX", "XY", "YY"]

def test_steane_code():
    n = 8

    steane_logicals = [[0, 1, 4], [0, 3, 6], [4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

    sim = QECSimulator(n)

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

    logical_z_qubits = steane_logicals[0]
    for qubit in logical_z_qubits:
        sim.measure(qubit)

    output = sim.measure_all(logical_z_qubits)

    circ = QuantumCircuit(n, n)

    circ.h(0)
    circ.h(4)
    circ.h(6)
    circ.cx(0, 1)
    circ.cx(4, 5)
    circ.cx(6, 3)
    circ.cx(4, 2)
    circ.cx(6, 5)
    circ.cx(0, 3)
    circ.cx(4, 1)
    circ.cx(3, 2)

    circ.cx(1, 7)
    circ.cx(3, 7)
    circ.cx(5, 7)

    circ.save_density_matrix()

    p0 = 1e-3

    p_base_1q = np.ones(3) * p0
    errors_1q = np.hstack((1 - np.sum(p_base_1q), p_base_1q))
    p_base_2q = np.ones(15) * p0
    errors_2q = np.hstack((1 - np.sum(p_base_2q), p_base_2q))

    stim_errors_1q = [p_base_1q[span_error_ordering_1q.index(stim_error_ordering_1q[i])] for i in range(3)]
    stim_errors_2q = [p_base_2q[span_error_ordering_2q.index(stim_error_ordering_2q[i])] for i in range(15)]

    measurement_results = np.zeros(sim.n_m)
    for i in range(1, sim.n_m):
        measurement_results[i] = None
    prob_0 = sim.get_marginal_probabilities(errors_1q, errors_2q, measurement_results=measurement_results)

    exp_val_z_free = sim.evaluate_expectation_value(output, errors_1q, errors_2q, measurement_results=None, normalise_by_trace=True)
    print()
    exp_val_z_postselected = sim.evaluate_expectation_value(output, errors_1q, errors_2q, measurement_results=measurement_results, normalise_by_trace=True)
    print()
    print(exp_val_z_postselected)

    error_1 = pauli_error([(pauli_strings_1q[i], errors_1q[i]) for i in range(4)])
    error_2 = pauli_error([(pauli_strings_2q[i], errors_2q[i]) for i in range(16)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    qiskit_prob_ancilla = rho.probabilities(qargs=[7])[0]

    circuit = stim.Circuit()

    circuit.append("H", 0)
    circuit.append("PAULI_CHANNEL_1", [0], stim_errors_1q)
    circuit.append("H", 4)
    circuit.append("PAULI_CHANNEL_1", [4], stim_errors_1q)
    circuit.append("H", 6)
    circuit.append("PAULI_CHANNEL_1", [6], stim_errors_1q)

    circuit.append("CNOT", [0, 1])
    circuit.append("PAULI_CHANNEL_2", [0, 1], stim_errors_2q)
    circuit.append("CNOT", [4, 5])
    circuit.append("PAULI_CHANNEL_2", [4, 5], stim_errors_2q)
    circuit.append("CNOT", [6, 3])
    circuit.append("PAULI_CHANNEL_2", [6, 3], stim_errors_2q)
    circuit.append("CNOT", [4, 2])
    circuit.append("PAULI_CHANNEL_2", [4, 2], stim_errors_2q)
    circuit.append("CNOT", [6, 5])
    circuit.append("PAULI_CHANNEL_2", [6, 5], stim_errors_2q)
    circuit.append("CNOT", [0, 3])
    circuit.append("PAULI_CHANNEL_2", [0, 3], stim_errors_2q)
    circuit.append("CNOT", [4, 1])
    circuit.append("PAULI_CHANNEL_2", [4, 1], stim_errors_2q)
    circuit.append("CNOT", [3, 2])
    circuit.append("PAULI_CHANNEL_2", [3, 2], stim_errors_2q)

    circuit.append("CNOT", [1, 7])
    circuit.append("PAULI_CHANNEL_2", [1, 7], stim_errors_2q)
    circuit.append("CNOT", [3, 7])
    circuit.append("PAULI_CHANNEL_2", [3, 7], stim_errors_2q)
    circuit.append("CNOT", [5, 7])
    circuit.append("PAULI_CHANNEL_2", [5, 7], stim_errors_2q)

    for i in logical_z_qubits + [7]:
        circuit.append("M", i)

    shots = 1_000_000
    sampler = circuit.compile_sampler()
    samples = sampler.sample(shots=shots)

    exp_val_qiskit = rho.expectation_value(Pauli("".join(["Z" if i in logical_z_qubits else "I" for i in range(n - 1, -1, -1)])))
    
    postselected_samples = samples[samples[:, -1] == 0, :-1]
    parities_postselected = np.sum(postselected_samples, axis=1) % 2
    exp_val_stim_postselected = np.average(1 - 2 * parities_postselected)

    assert np.isclose(exp_val_z_free, exp_val_qiskit)
    assert np.isclose(exp_val_z_postselected, exp_val_stim_postselected, atol=0.01)
    assert np.isclose(prob_0, qiskit_prob_ancilla)
