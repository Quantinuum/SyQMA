import numpy as np
import pytest
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from syqma import Simulator


def test_span():
    sim = Simulator(2)

    assert np.allclose(np.hstack((sim.span_x, sim.span_z)), np.array([[0, 0, 1, 0], [0, 0, 0, 1]]))

    sim.h(0)
    sim.cnot(0, 1)

    assert np.allclose(np.hstack((sim.span_x, sim.span_z)), np.array([[1, 1, 0, 0], [0, 0, 1, 1]]))


pauli_strings_1q = ["I", "Z", "X", "Y"]
# inverse endianness in Qiskit
pauli_strings_2q = ["II", "ZI", "IZ", "ZZ", "XI", "YI", "XZ", "YZ", "IX", "ZX", "IY", "ZY", "XX", "YX", "XY", "YY"]
np.random.seed(0)
p0 = 1e-3
p_base_1q = np.ones(3) * p0
errors_1q = np.hstack((1 - np.sum(p_base_1q), p_base_1q))
p_base_2q = np.random.rand(15) * p0
errors_2q = np.hstack((1 - np.sum(p_base_2q), p_base_2q))
error_1 = pauli_error([(pauli_strings_1q[i], errors_1q[i]) for i in range(4)])
error_2 = pauli_error([(pauli_strings_2q[i], errors_2q[i]) for i in range(16)])

print(f"p_base_1q = {p_base_1q}")
print(f"min = {np.min(p_base_1q)}, max = {np.max(p_base_1q)}, mean = {np.mean(p_base_1q)}")
print(f"p_base_2q = {p_base_2q}")
print(f"min = {np.min(p_base_2q)}, max = {np.max(p_base_2q)}, mean = {np.mean(p_base_2q)}")

all_n = [1, 2, 3, 4]
all_fold = [0, 1, 2]


@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("fold", all_fold)
def test_equal_qiskit_noiseless_global(n: int, fold: int):
    simulator = AerSimulator(method="density_matrix")
    circ = QuantumCircuit(n)

    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    for _ in range(fold):
        circ.h(0)
        for i in range(n - 1):
            circ.cx(i, i + 1)
        for i in range(n - 1, 0, -1):
            circ.cx(i - 1, i)
        circ.h(0)

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    sim.h(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)
    for _ in range(fold):
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        for i in range(n - 1, 0, -1):
            sim.cnot(i - 1, i)
        sim.h(0)

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)

    assert np.allclose(exp_val, exp_val_qiskit)


@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("fold", all_fold)
def test_equal_qiskit_noiseless_local(n: int, fold: int):
    simulator = AerSimulator(method="density_matrix")
    circ = QuantumCircuit(n)

    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    for _ in range(fold):
        circ.h(0)
        for i in range(n - 1):
            circ.cx(i, i + 1)
        for i in range(n - 1, 0, -1):
            circ.cx(i - 1, i)
        circ.h(0)

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" + "I" * (n - 1)))

    sim = Simulator(n)

    sim.h(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)
    for _ in range(fold):
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        for i in range(n - 1, 0, -1):
            sim.cnot(i - 1, i)
        sim.h(0)

    measured_qubits = np.array([0])
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)

    assert np.allclose(exp_val, exp_val_qiskit)


@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("fold", all_fold)
def test_equal_qiskit_noisy_global_1q(n: int, fold: int):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    for _ in range(fold):
        circ.h(0)
        for i in range(n - 1):
            circ.cx(i, i + 1)
        for i in range(n - 1, 0, -1):
            circ.cx(i - 1, i)
        circ.h(0)

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    sim.h(0)
    sim.pauli_error_1(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)
    for _ in range(fold):
        sim.h(0)
        sim.pauli_error_1(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
        for i in range(n - 1, 0, -1):
            sim.cnot(i - 1, i)
        sim.h(0)
        sim.pauli_error_1(0)

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)

    assert np.allclose(exp_val, exp_val_qiskit)


@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("fold", all_fold)
def test_equal_qiskit_noisy_global_2q(n: int, fold: int):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    for _ in range(fold):
        circ.h(0)
        for i in range(n - 1):
            circ.cx(i, i + 1)
        for i in range(n - 1, 0, -1):
            circ.cx(i - 1, i)
        circ.h(0)

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    sim.h(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for _ in range(fold):
        sim.h(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
            sim.pauli_error_2((i, i + 1))
        for i in range(n - 1, 0, -1):
            sim.cnot(i - 1, i)
            sim.pauli_error_2((i - 1, i))
        sim.h(0)

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)

    assert np.allclose(exp_val, exp_val_qiskit)


@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("fold", all_fold)
def test_equal_qiskit_noisy_global_mixed(n: int, fold: int):
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    for _ in range(fold):
        circ.h(0)
        for i in range(n - 1):
            circ.cx(i, i + 1)
        for i in range(n - 1, 0, -1):
            circ.cx(i - 1, i)
        circ.h(0)

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    sim.h(0)
    sim.pauli_error_1(0)
    for i in range(n - 1):
        sim.cnot(i, i + 1)
        sim.pauli_error_2((i, i + 1))
    for _ in range(fold):
        sim.h(0)
        sim.pauli_error_1(0)
        for i in range(n - 1):
            sim.cnot(i, i + 1)
            sim.pauli_error_2((i, i + 1))
        for i in range(n - 1, 0, -1):
            sim.cnot(i - 1, i)
            sim.pauli_error_2((i - 1, i))
        sim.h(0)
        sim.pauli_error_1(0)

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)

    print()
    print(exp_val)
    print(exp_val_qiskit)

    assert np.allclose(exp_val, exp_val_qiskit)

all_n = [2, 3, 4]
all_n_gates = [10, 20, 50]

@pytest.mark.parametrize("n", all_n)
@pytest.mark.parametrize("n_gates", all_n_gates)
def test_equal_qiskit_random_multi(n: int, n_gates: int):
    gates = random.choices(["h", "cx"], k=n_gates)
    qubits = random.choices(range(n), k=2 * n_gates)
    
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    for i in range(n_gates):
        if gates[i] == "h":
            circ.h(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                circ.h(qubits[2 * i])
            else:
                circ.cx(qubits[2 * i], qubits[2 * i + 1])

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    for i in range(n_gates):
        if gates[i] == "h":
            sim.h(qubits[2 * i])
            sim.pauli_error_1(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                sim.h(qubits[2 * i])
                sim.pauli_error_1(qubits[2 * i])
            else:
                sim.cnot(qubits[2 * i], qubits[2 * i + 1])
                sim.pauli_error_2((qubits[2 * i], qubits[2 * i + 1]))

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)
    
    assert np.allclose(exp_val, exp_val_qiskit)


def test_equal_qiskit_random_single_1():
    n = 2
    n_gates = 20
    gates = ['cx', 'cx', 'h', 'cx', 'h', 'cx', 'cx', 'cx', 'h', 'h', 'h', 'h', 'h', 'cx', 'h', 'cx', 'cx', 'h', 'h', 'cx']
    qubits = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]
  
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    for i in range(n_gates):
        if gates[i] == "h":
            circ.h(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                circ.h(qubits[2 * i])
            else:
                circ.cx(qubits[2 * i], qubits[2 * i + 1])

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    for i in range(n_gates):
        if gates[i] == "h":
            sim.h(qubits[2 * i])
            sim.pauli_error_1(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                sim.h(qubits[2 * i])
                sim.pauli_error_1(qubits[2 * i])
            else:
                sim.cnot(qubits[2 * i], qubits[2 * i + 1])
                sim.pauli_error_2((qubits[2 * i], qubits[2 * i + 1]))

    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)
    
    assert np.allclose(exp_val, exp_val_qiskit)
    

def test_equal_qiskit_random_single_2():
    n = 4
    n_gates = 50
    gates = ['cx', 'cx', 'cx', 'cx', 'h', 'cx', 'h', 'cx', 'cx', 'cx', 'h', 'cx', 'cx', 'cx', 'cx', 'h', 'cx', 'h', 'h', 'cx', 'h', 'h', 'cx', 'cx', 'h', 'cx', 'cx', 'h', 'h', 'h', 'h', 'h', 'cx', 'cx', 'cx', 'cx', 'cx', 'cx', 'cx', 'h', 'h', 'h', 'cx', 'h', 'cx', 'cx', 'cx', 'cx', 'h', 'h']
    qubits = [2, 2, 2, 2, 1, 2, 0, 1, 0, 3, 0, 3, 0, 0, 0, 1, 3, 0, 2, 3, 2, 2, 3, 3, 0, 2, 3, 3, 3, 3, 0, 2, 2, 0, 3, 1, 3, 1, 3, 0, 1, 2, 0, 0, 1, 1, 1, 0, 2, 3, 1, 3, 0, 2, 1, 0, 0, 0, 3, 3, 1, 3, 2, 3, 1, 0, 2, 2, 2, 2, 1, 1, 1, 2, 0, 3, 0, 0, 3, 3, 0, 1, 0, 2, 0, 2, 1, 0, 3, 3, 2, 0, 3, 2, 0, 3, 3, 0, 2, 1]
  
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ["h"])
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    simulator = AerSimulator(method="density_matrix", noise_model=noise_model)
    circ = QuantumCircuit(n)

    for i in range(n_gates):
        if gates[i] == "h":
            circ.h(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                circ.h(qubits[2 * i])
            else:
                circ.cx(qubits[2 * i], qubits[2 * i + 1])

    circ.save_density_matrix()

    result = simulator.run(circ).result()
    rho = result.to_dict()["results"][0]["data"]["density_matrix"]

    exp_val_qiskit = rho.expectation_value(Pauli("Z" * n))

    sim = Simulator(n)

    for i in range(n_gates):
        if gates[i] == "h":
            sim.h(qubits[2 * i])
            sim.pauli_error_1(qubits[2 * i])
        else:
            if qubits[2 * i] == qubits[2 * i + 1]:
                sim.h(qubits[2 * i])
                sim.pauli_error_1(qubits[2 * i])
            else:
                sim.cnot(qubits[2 * i], qubits[2 * i + 1])
                sim.pauli_error_2((qubits[2 * i], qubits[2 * i + 1]))
    
    measured_qubits = np.array(list(range(n)))
    exp_val = sim.evaluate_expectation_value(sim.measure_all(measured_qubits), errors_1q, errors_2q)
    
    assert np.allclose(exp_val, exp_val_qiskit)
