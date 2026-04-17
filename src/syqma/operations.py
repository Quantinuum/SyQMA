"""State-span and gate operations for the SyQMA tableau representation."""

import math
from collections.abc import Callable

import numpy as np
from .pauli import commutation, generate_paulis, pauli_string_to_symplectic
from sympy.discrete.transforms import fwht

DISPLAY_STRINGS = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "h": "H",
    "s": "S",
    "sdag": "S†",
    "reset": "R",
    "project_on_0": "|0><0|",
    "measure": "M",
    "cnot": "@X",
    "ch": "@H",
}


class Span:
    """Represent the symbolic stabiliser span for a circuit."""

    def __init__(self, n: int = 0) -> None:
        """Initialise the span.

        Args:
            n (int): The number of qubits in the circuit.

        """
        self.n_qubits = n
        self.n_deterministic_measurements = 0
        self.n_magic = 0
        self.n_stabs = self.n_qubits + self.n_deterministic_measurements + self.n_magic
        self.phases = np.zeros(self.n_stabs, dtype=np.uint8)
        self.n_m = 0
        self.output_trace = None
        self.symbolic_phases = np.zeros((self.n_stabs, 0), dtype=np.uint8)
        self.errors = []
        self.error_indices_list = []
        self.p_to_q_dict = {}

        self.c_operators = np.zeros((self.n_stabs, 0), dtype=np.uint8)
        self.s_operators = np.zeros((self.n_stabs, 0), dtype=np.uint8)
        self.o_operators = np.zeros((self.n_stabs, 0), dtype=np.uint8)
        self.theta_indices = np.zeros(0, dtype=np.uint8)

        self.span_x = np.zeros((self.n_stabs, self.n_qubits), dtype=np.uint8, order="F")
        self.span_z = np.zeros((self.n_stabs, self.n_qubits), dtype=np.uint8, order="F")
        for i in range(self.n_stabs):
            self.span_z[i, i] = 1

        # qubits become inactive when measured (maybe just setting to 0 is enough)
        self.next_fresh_qubit = self.n_qubits
        self.active_qubits = np.ones(self.n_qubits, dtype=bool)

        self.list_error_channels: list[int] = []

        self.non_deterministic_measurements = []
        self.measurement_qubits = []

        (
            self.all_pauli_indices_1q,
            self.all_pauli_indices_1q_reverse,
            self.all_pauli_indices_2q,
            self.all_pauli_indices_2q_reverse,
            self.pauli_multiplication_dict_1q,
            self.pauli_multiplication_dict_2q,
        ) = generate_paulis()

        self.create_circuit = True
        self.gates = []

    def symplectic_span(self) -> np.ndarray:
        """Return the concatenated XZ symplectic span."""
        return np.hstack((self.span_x, self.span_z))

    def pauli_span(self) -> str:
        """Return a visual Pauli-product representation of the span.

        Returns:
            str: Visual representation of the tableau of stabilisers.

        """

        def _cell(row: int, col: int) -> str:
            k = int(self.span_x[row, col]) + 2 * int(self.span_z[row, col])

            return [".", "X", "Z", "Y"][k]

        def _row(row: int) -> str:
            result = ""
            for col in range(self.n_qubits):
                result += str(_cell(row, col))

            return result

        z_obs = [_row(row) for row in range(self.n_stabs)]

        return "\n".join(z_obs)

    def draw_circuit(
        self, include_noise: bool = True, gate_padding: int = 3, error_padding: int = 1
    ) -> str:
        """Draw the circuit with a qubit on each line.

        Returns:
            str: The circuit in a string format.

        """
        str_circuit = [f"{' ' * math.floor(math.log10(self.n_qubits - 1))}0: "]
        temp = [
            f"{' ' * (math.floor(math.log10(self.n_qubits - 1)) - math.floor(math.log10(i // 2)))}{i // 2}: "
            if i % 2 == 0
            else f"{' ' * math.floor(math.log10(self.n_qubits - 1))}      "
            for i in range(1, 2 * self.n_qubits - 1)
        ]
        str_circuit += temp

        for gate in self.gates:
            if gate[0] in [
                "x",
                "y",
                "z",
                "h",
                "s",
                "sdag",
                "reset",
                "project_on_0",
                "measure",
            ]:
                j = gate[1][0]
                display_string = DISPLAY_STRINGS[gate[0]]
                str_circuit[2 * j] += f"{'-' * gate_padding}{display_string}"
            elif gate[0] in ["rz", "ry"]:
                j = gate[1][0]
                theta_index = gate[1][1]
                str_circuit[2 * j] += (
                    f"{'-' * gate_padding}R{gate[0][1].upper()}(\u03b8_{theta_index})"
                )
            elif gate[0] in ["cnot", "ch", "mpp"]:
                if gate[0] in DISPLAY_STRINGS:
                    if gate[0] == "ch":
                        qubits = gate[1][:2]
                    else:
                        qubits = gate[1]
                    display_string = DISPLAY_STRINGS[gate[0]]
                else:
                    qubits = gate[1][0]
                    display_string = gate[1][1]
                min_idx = min(qubits)
                max_idx = max(qubits)
                max_length = max(
                    [len(str_circuit[i]) for i in range(2 * min_idx, 2 * max_idx + 1)]
                )
                for i, q in enumerate(qubits):
                    str_circuit[2 * q] += (
                        f"{'-' * max(0, (max_length - len(str_circuit[2 * q])))}{'-' * gate_padding}{display_string[i]}"
                    )
                for i in range(2 * min_idx + 1, 2 * max_idx):
                    if i % 2 == 0:
                        if i // 2 not in qubits:
                            str_circuit[i] += (
                                f"{'-' * max(0, (max_length - len(str_circuit[i])))}{'-' * gate_padding}|"
                            )
                    else:
                        str_circuit[i] += (
                            f"{' ' * max(0, (max_length - len(str_circuit[i])))}{' ' * gate_padding}|"
                        )
            elif gate[0] == "pauli_error_1" and include_noise:
                j = gate[1][0]
                str_circuit[2 * j] += f"{'-' * error_padding}E"
            elif gate[0] == "pauli_error_2" and include_noise:
                j = gate[1][0][0]
                k = gate[1][0][1]
                min_idx = min(j, k)
                max_idx = max(j, k)
                max_length = max(
                    [len(str_circuit[i]) for i in range(2 * min_idx, 2 * max_idx + 1)]
                )
                str_circuit[2 * j] += (
                    f"{'-' * max(0, (max_length - len(str_circuit[2 * j])))}{'-' * error_padding}E"
                )
                str_circuit[2 * k] += (
                    f"{'-' * max(0, (max_length - len(str_circuit[2 * k])))}{'-' * error_padding}E"
                )
                for i in range(2 * min_idx + 1, 2 * max_idx):
                    if i % 2 == 0:
                        str_circuit[i] += (
                            f"{'-' * max(0, (max_length - len(str_circuit[i])))}{'-' * error_padding}|"
                        )
                    else:
                        str_circuit[i] += (
                            f"{' ' * max(0, (max_length - len(str_circuit[i])))}{' ' * error_padding}|"
                        )

        total_length = max([len(line) for line in str_circuit]) + gate_padding
        final_str_circuit = []
        for i, line in enumerate(str_circuit):
            if i % 2 == 0:
                final_str_circuit.append(line + "-" * (total_length - len(line)))
            else:
                final_str_circuit.append(line + " " * (total_length - len(line)))

        return "\n".join(final_str_circuit)

    def initialise_symbolic_stabiliser(
        self, j: int | list[int] | np.ndarray, np_pauli: np.ndarray = None
    ) -> None:
        """Add a symbolic stabiliser row for a measurement outcome."""
        new_phases = np.zeros(self.n_stabs + 1, dtype=int)
        new_phases[: self.n_stabs] = self.phases[: self.n_stabs]
        self.phases = new_phases
        # ! a bit hacky, should recheck
        new_symbolic_phases = np.zeros(
            (self.n_stabs + 1, self.symbolic_phases.shape[1] + 1), dtype=int
        )
        new_symbolic_phases[: self.n_stabs, : self.symbolic_phases.shape[1]] = (
            self.symbolic_phases[: self.n_stabs]
        )
        self.symbolic_phases = new_symbolic_phases

        new_span_x = np.zeros((self.n_stabs + 1, self.n_qubits), dtype=int, order="F")
        new_span_z = np.zeros((self.n_stabs + 1, self.n_qubits), dtype=int, order="F")
        new_span_x[: self.n_stabs] = self.span_x[: self.n_stabs]
        new_span_z[: self.n_stabs] = self.span_z[: self.n_stabs]

        if isinstance(j, int) and np_pauli is None:
            new_span_z[self.n_stabs, j] = 1
        else:
            new_span_x[self.n_stabs, j] = np_pauli[: np_pauli.shape[0] // 2]
            new_span_z[self.n_stabs, j] = np_pauli[np_pauli.shape[0] // 2 :]
        self.span_x = new_span_x
        self.span_z = new_span_z

        new_c_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_c_operators[: self.n_stabs, : self.c_operators.shape[1]] = self.c_operators
        self.c_operators = new_c_operators
        new_s_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_s_operators[: self.n_stabs, : self.s_operators.shape[1]] = self.s_operators
        self.s_operators = new_s_operators
        new_o_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_o_operators[: self.n_stabs, : self.o_operators.shape[1]] = self.o_operators
        self.o_operators = new_o_operators

        self.n_deterministic_measurements += 1
        self.n_stabs += 1

    def initialise_qubit(self, np_pauli: np.ndarray = None) -> None:
        """Add a fresh qubit row to the span."""
        new_phases = np.zeros(self.n_stabs + 1, dtype=int)
        new_phases[: self.n_stabs] = self.phases[: self.n_stabs]
        self.phases = new_phases
        new_symbolic_phases = np.zeros(
            (self.n_stabs + 1, self.symbolic_phases.shape[1]), dtype=int
        )
        new_symbolic_phases[: self.n_stabs] = self.symbolic_phases[: self.n_stabs]
        self.symbolic_phases = new_symbolic_phases

        new_span_x = np.zeros(
            (self.n_stabs + 1, self.n_qubits + 1), dtype=int, order="F"
        )
        new_span_z = np.zeros(
            (self.n_stabs + 1, self.n_qubits + 1), dtype=int, order="F"
        )
        new_span_x[: self.n_stabs, : self.n_qubits] = self.span_x[
            : self.n_stabs, : self.n_qubits
        ]
        new_span_z[: self.n_stabs, : self.n_qubits] = self.span_z[
            : self.n_stabs, : self.n_qubits
        ]
        if np_pauli is None:
            new_span_z[self.n_stabs, self.n_qubits] = 1
        else:
            new_span_x[self.n_stabs, : self.n_qubits] = np_pauli[
                : np_pauli.shape[0] // 2
            ]
            new_span_z[self.n_stabs, : self.n_qubits] = np_pauli[
                np_pauli.shape[0] // 2 :
            ]
        self.span_x = new_span_x
        self.span_z = new_span_z

        new_c_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_c_operators[: self.n_stabs, : self.c_operators.shape[1]] = self.c_operators
        self.c_operators = new_c_operators
        new_s_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_s_operators[: self.n_stabs, : self.s_operators.shape[1]] = self.s_operators
        self.s_operators = new_s_operators
        new_o_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_o_operators[: self.n_stabs, : self.o_operators.shape[1]] = self.o_operators
        self.o_operators = new_o_operators

        self.n_qubits += 1
        self.n_stabs += 1

    def initialise_magic_stabiliser(self, j: int) -> None:
        """Add a stabiliser row for a non-Clifford rotation branch."""
        new_phases = np.zeros(self.n_stabs + 1, dtype=int)
        new_phases[: self.n_stabs] = self.phases[: self.n_stabs]
        self.phases = new_phases
        new_symbolic_phases = np.zeros(
            (self.n_stabs + 1, self.symbolic_phases.shape[1]), dtype=int
        )
        new_symbolic_phases[: self.n_stabs] = self.symbolic_phases[: self.n_stabs]
        self.symbolic_phases = new_symbolic_phases

        new_span_x = np.zeros((self.n_stabs + 1, self.n_qubits), dtype=int, order="F")
        new_span_z = np.zeros((self.n_stabs + 1, self.n_qubits), dtype=int, order="F")
        new_span_x[: self.n_stabs] = self.span_x[: self.n_stabs]
        new_span_z[: self.n_stabs] = self.span_z[: self.n_stabs]
        new_span_z[self.n_stabs, j] = 1
        self.span_x = new_span_x
        self.span_z = new_span_z

        self.n_magic += 1

        new_c_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_c_operators[: self.n_stabs, : self.c_operators.shape[1]] = self.c_operators[
            : self.n_stabs
        ]
        self.c_operators = new_c_operators
        new_s_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_s_operators[: self.n_stabs, : self.s_operators.shape[1]] = self.s_operators[
            : self.n_stabs
        ]
        self.s_operators = new_s_operators
        new_o_operators = np.zeros((self.n_stabs + 1, self.n_magic), dtype=int)
        new_o_operators[: self.n_stabs, : self.o_operators.shape[1]] = self.o_operators[
            : self.n_stabs
        ]
        self.o_operators = new_o_operators

        self.n_stabs += 1

    def add_gate(func: Callable) -> Callable:
        """Record a method call in the circuit before applying it."""

        def wrapper(self, *args, **kwargs) -> object:
            if self.create_circuit:
                self.gates.append((func.__name__, args, kwargs))
            return func(self, *args, **kwargs)

        return wrapper

    @add_gate
    def x(self, j: int) -> None:
        """Apply a Pauli X gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.phases ^= self.span_z[:, j]

    @add_gate
    def y(self, j: int) -> None:
        """Apply a Pauli Y gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.phases ^= self.span_x[:, j] ^ self.span_z[:, j]

    @add_gate
    def z(self, j: int) -> None:
        """Apply a Pauli Z gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.phases ^= self.span_x[:, j]

    @add_gate
    def h(self, j: int) -> None:
        """Apply a Hadamard gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.phases ^= self.span_x[:, j] & self.span_z[:, j]
        self.span_x[:, j], self.span_z[:, j] = (
            self.span_z[:, j].copy(),
            self.span_x[:, j].copy(),
        )

    @add_gate
    def s(self, j: int) -> None:
        """Apply a phase gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.phases ^= self.span_x[:, j] & self.span_z[:, j]
        self.span_z[:, j] ^= self.span_x[:, j]

    @add_gate
    def sdag(self, j: int) -> None:
        """Apply the inverse phase gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.

        """
        self.s(j)
        self.s(j)
        self.s(j)

    @add_gate
    def cnot(self, j: int, k: int) -> None:
        """Apply a CNOT gate from the j-th qubit to the k-th qubit.

        Args:
            j (int): Index of the control qubit.
            k (int): Index of the target qubit.

        """
        self.phases ^= (
            self.span_x[:, j]
            & self.span_z[:, k]
            & (self.span_x[:, k] ^ self.span_z[:, j] ^ 1)
        )
        self.span_z[:, j] ^= self.span_z[:, k]
        self.span_x[:, k] ^= self.span_x[:, j]

    @add_gate
    def rz(self, j: int, theta_index: int) -> None:
        """Apply an RZ(theta_index) gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.
            theta_index (int): Index of the angle.

        """
        anticommutes: bool = False
        for i in range(self.span_x.shape[0]):
            if self.span_x[i, j] == 1:
                anticommutes = True
        if anticommutes:
            self.initialise_magic_stabiliser(j)
            for i in range(self.span_x.shape[0]):
                if self.span_x[i, j] == 1:
                    self.c_operators[i, self.n_magic - 1] ^= 1
            new_theta_indices = np.zeros(self.n_magic, dtype=int)
            new_theta_indices[: self.n_magic - 1] = self.theta_indices
            new_theta_indices[self.n_magic - 1] = theta_index
            self.theta_indices = new_theta_indices
            self.o_operators[self.n_stabs - 1, self.n_magic - 1] = 1

    @add_gate
    def ry(self, j: int, theta_index: int) -> None:
        """Apply an RY(theta_index) gate to the j-th qubit.

        Args:
            j (int): Index of the qubit.
            theta_index (int): Index of the angle.

        """
        self.s(j)
        self.h(j)
        self.rz(j, theta_index)
        self.h(j)
        self.sdag(j)

    @add_gate
    def ch(self, j: int, k: int, theta_index: int, noise: bool = True) -> None:
        """Apply a controlled-H construction using indexed rotations.

        Args:
            j (int): Index of the control qubit.
            k (int): Index of the target qubit.
            theta_index (int): Index of the angle.
            noise (bool): Whether to add Pauli noise after decomposed gates.

        """
        # TODO decide what to do about the noise
        self.ry(k, theta_index + 1)
        if noise:
            self.pauli_error_1(k)

        self.h(k)
        if noise:
            self.pauli_error_1(k)
        self.cnot(j, k)
        if noise:
            self.pauli_error_2((j, k))
        self.h(k)
        if noise:
            self.pauli_error_1(k)

        self.ry(k, theta_index)
        if noise:
            self.pauli_error_1(k)

    @add_gate
    def measure(self, q: int, trace_out: bool = False) -> None:
        """Measure one qubit in the Z basis."""
        if self.active_qubits[q] is False:
            raise ValueError("Qubit was already measured and not re-initialised.")

        p = -1
        for i in range(self.n_stabs):
            if self.span_x[i][q] == 1:
                p = i
                break

        if p > -1:
            self.non_deterministic_measurements.append(self.n_stabs)

        if trace_out is False:
            self.initialise_symbolic_stabiliser(q)
            self.symbolic_phases[self.n_stabs - 1, self.n_m] = 1
        else:
            # ! this doesn't work for final measurements now though
            # ! projection + tracing out
            self.span_z[:, q] ^= 1

        self.n_m += 1
        self.measurement_qubits.append([q])

    @add_gate
    def mpp(
        self, measured_qubits: list[int], pauli_string: str, trace_out: bool = False
    ) -> None:
        """Measure a multi-qubit Pauli product."""
        if any(self.active_qubits[q] is False for q in measured_qubits):
            raise ValueError("Qubit was already measured and not re-initialised.")

        np_pauli = pauli_string_to_symplectic(pauli_string)

        measured_qubits = np.array(measured_qubits)
        span_subset = np.hstack(
            (self.span_x[:, measured_qubits], self.span_z[:, measured_qubits])
        )

        p = -1
        for i in range(self.n_stabs):
            if commutation(span_subset[i], np_pauli) == 1:
                p = i
                break

        if p > -1:
            print(
                f"\nWARNING: measurement of qubits {measured_qubits} is non-deterministic\n"
            )
            self.non_deterministic_measurements.append(self.n_stabs)

        if trace_out is False:
            self.initialise_symbolic_stabiliser(measured_qubits, np_pauli)
            self.symbolic_phases[self.n_stabs - 1, self.n_m] = 1

        self.n_m += 1

        self.measurement_qubits.append(measured_qubits)

    @add_gate
    def reset(self, q: int, pauli_string: str = "Z") -> None:
        """Reset a qubit into the requested Pauli basis.

        Args:
            q (int): The qubit to reset.
            pauli_string (str): Pauli basis for the reset state.

        """
        if pauli_string == "Z":
            pauli_array = np.array([0, 1])
        elif pauli_string == "X":
            pauli_array = np.array([1, 0])
        elif pauli_string == "Y":
            pauli_array = np.array([1, 1])
        else:
            raise ValueError(f"Unsupported Pauli string: {pauli_string}")

        self.initialise_qubit(pauli_array)
        self.span_x[:, q], self.span_x[:, self.n_qubits - 1] = (
            self.span_x[:, self.n_qubits - 1],
            self.span_x[:, q],
        )
        self.span_z[:, q], self.span_z[:, self.n_qubits - 1] = (
            self.span_z[:, self.n_qubits - 1],
            self.span_z[:, q],
        )

    @add_gate
    def pauli_error_1(self, k: int) -> None:
        """Apply a one-qubit Pauli channel to the k-th qubit.

        Args:
            k (int): Index of the qubit.

        """
        span_subset = tuple((self.span_x[:, k].T, self.span_z[:, k].T))
        pauli_indices = self.all_pauli_indices_1q[span_subset]

        self.error_indices_list.append(pauli_indices)
        self.list_error_channels.append(1)

    @add_gate
    def pauli_error_2(self, qubit_pair: tuple[int, int]) -> None:
        """Apply a two-qubit Pauli channel to a qubit pair.

        Args:
            qubit_pair (Tuple[int]): Pair of (control, target) qubits.

        """
        q_1, q_2 = qubit_pair
        span_subset = tuple(
            (
                self.span_x[:, q_1].T,
                self.span_x[:, q_2].T,
                self.span_z[:, q_1].T,
                self.span_z[:, q_2].T,
            )
        )
        pauli_indices = self.all_pauli_indices_2q[span_subset]

        self.error_indices_list.append(pauli_indices)
        self.list_error_channels.append(2)

    def get_error_coefficients(
        self, errors_1q: list[float], errors_2q: list[float]
    ) -> np.ndarray:
        """Calculate error coefficients for the current error channels.

        Args:
            errors_1q (list[float]): The error rates for single-qubit errors.
            errors_2q (list[float]): The error rates for two-qubit errors.

        Returns:
            np.ndarray: The error coefficients for the given error channels.

        """
        error_channels = [
            errors_1q if error_channel == 1 else errors_2q
            for error_channel in self.list_error_channels
        ]
        error_coefficients = np.zeros((len(self.error_indices_list), 16), dtype=float)

        # swap X and Z to compute symplectic kernel instead of dot-product kernel
        perm_1q_symp = np.array([0, 2, 1, 3], dtype=int)
        perm_2q_symp = np.array(
            [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], dtype=int
        )

        for i, (chan_kind, p_0) in enumerate(
            zip(self.list_error_channels, error_channels)
        ):
            p_0_tuple = tuple(p_0)
            if p_0_tuple in self.p_to_q_dict:
                p_1 = self.p_to_q_dict[p_0_tuple]
            else:
                p_0 = np.asarray(p_0)
                if chan_kind == 1 and p_0.size == 4:
                    p_0 = p_0[perm_1q_symp]
                elif chan_kind == 2 and p_0.size == 16:
                    p_0 = p_0[perm_2q_symp]

                p_1 = fwht(p_0)
                self.p_to_q_dict[p_0_tuple] = p_1

            error_coefficients[i, : p_0.size] = p_1

        return error_coefficients
