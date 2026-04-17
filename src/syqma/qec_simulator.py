"""QEC-focused convenience methods built on the base SyQMA simulator."""

import itertools
import numpy as np

from .simulator import ErrorRates, Simulator


class QECSimulator(Simulator):
    """Add QEC postselection and lookup-table helpers to ``Simulator``."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialise the QEC simulator.

        Args:
            *args: The arguments to pass to the parent class.
            **kwargs: The keyword arguments to pass to the parent class.

        """
        super().__init__(*args, **kwargs)

    def copy(self) -> "QECSimulator":
        """Return a shallow copy of the simulator state."""
        sim = QECSimulator(n=self.n_qubits)

        sim.gates = self.gates
        sim.n_qubits = self.n_qubits
        sim.n_deterministic_measurements = self.n_deterministic_measurements
        sim.n_magic = self.n_magic
        sim.n_stabs = self.n_stabs
        sim.n_m = self.n_m
        sim.output_trace = self.output_trace

        sim.phases = self.phases
        sim.symbolic_phases = self.symbolic_phases
        sim.errors = self.errors
        sim.error_indices = self.error_indices
        sim.p_to_q_dict = self.p_to_q_dict

        sim.c_operators = self.c_operators
        sim.s_operators = self.s_operators
        sim.o_operators = self.o_operators
        sim.theta_indices = self.theta_indices

        sim.all_simulators = self.all_simulators

        sim.span = self.span
        sim.list_error_channels = self.list_error_channels
        sim.finished_circuit = self.finished_circuit

        return sim

    def measure_stabilisers_virtually(
        self, stabilisers: list[tuple[str, list[int]]]
    ) -> None:
        """Add virtual multi-Pauli stabiliser measurements."""
        for stabiliser in stabilisers:
            pauli_string, qubits = stabiliser
            self.mpp(qubits, pauli_string)

    def acceptance_probability(
        self,
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        postselected_indices: list[int] = None,
        thetas: list[float] = None,
    ) -> float:
        """Return the probability that the postselected measurements pass."""
        measurement_results = np.zeros(self.n_m)
        measurement_results[np.setdiff1d(np.arange(self.n_m), postselected_indices)] = (
            None
        )

        return self.get_marginal_probabilities(
            errors_1q, errors_2q, measurement_results=measurement_results, thetas=thetas
        )

    def postselected_expectation_value(
        self,
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        measured_qubits: list[int],
        measured_pauli_string: str = None,
        postselected_indices: list[int] = None,
        thetas: list[float] = None,
        output: object = None,
    ) -> float | np.ndarray:
        """Return an expectation value conditioned on accepted measurements."""
        if measured_pauli_string is None:
            measured_pauli_string = "Z" * len(measured_qubits)

        if postselected_indices is None:
            measurement_results = None
        else:
            measurement_results = np.zeros(self.n_m)
            measurement_results[
                np.setdiff1d(np.arange(self.n_m), postselected_indices)
            ] = None

        if output is None:
            for i, q in enumerate(measured_qubits):
                if measured_pauli_string[i] == "X":
                    self.h(q)
                elif measured_pauli_string[i] == "Y":
                    self.s(q)
                    self.s(q)
                    self.s(q)
                    self.h(q)
            output = self.measure_all(measured_qubits)
            for i, q in enumerate(measured_qubits):
                if measured_pauli_string[i] == "X":
                    self.h(q)
                elif measured_pauli_string[i] == "Y":
                    self.h(q)
                    self.s(q)

        exp_val_postselected = self.evaluate_expectation_value(
            output,
            errors_1q,
            errors_2q,
            measurement_results=measurement_results,
            thetas=thetas,
            normalise_by_trace=True,
        )

        return exp_val_postselected

    def lut_exp_vals_from_syndromes_memoryless(
        self,
        all_errors_1q: list[ErrorRates],
        all_errors_2q: list[ErrorRates],
        measured_qubits: list[int],
        postselected_indices: list[int] = None,
        all_thetas: list[float] = [None],
        batch_size: int = 1,
        n_jobs: int = -1,
        device: str | None = None,
    ) -> tuple[list[dict[tuple[int, ...], float]], list[dict[tuple[int, ...], float]]]:
        """Build syndrome lookup tables without storing intermediate outputs."""
        # ! assumes postselected indices are at the beginning
        syndromes = [
            (0,) * len(postselected_indices) + syndrome
            for syndrome in itertools.product(
                [0, 1], repeat=self.n_m - len(postselected_indices)
            )
        ]
        all_measurement_results = np.array(syndromes)

        all_exp_vals, all_probs = self.evaluate_expectation_values_from_scratch(
            measured_qubits,
            all_errors_1q,
            all_errors_2q,
            all_measurement_results=all_measurement_results,
            all_thetas=all_thetas,
            postselected_indices=postselected_indices,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )

        all_exp_vals_dict = []

        for exp_vals in all_exp_vals[:, :, 0].T:
            exp_vals_dict = {
                syndrome[len(postselected_indices) :]: exp_val
                for syndrome, exp_val in zip(syndromes, exp_vals)
            }
            all_exp_vals_dict.append(exp_vals_dict)

        all_probs_dict = []

        for probs in all_probs[:, :, 0].T:
            probs_dict = {
                syndrome[len(postselected_indices) :]: prob
                for syndrome, prob in zip(syndromes, probs)
            }
            all_probs_dict.append(probs_dict)

        return all_exp_vals_dict, all_probs_dict

    def lut_exp_vals_from_syndromes(
        self,
        operator: tuple[str, list[int]],
        measurement_indices: list[int],
        errors_1q: ErrorRates,
        errors_2q: ErrorRates,
        thetas: list[float] = None,
        output: object = None,
        postselected_indices: list[int] = [],
        printing: bool = True,
    ) -> dict[tuple[int, ...], float]:
        """Build a syndrome-to-expectation-value lookup table."""
        # ! assumes postselected indices are at the beginning
        syndromes = [
            (0,) * len(postselected_indices) + syndrome
            for syndrome in itertools.product(
                [0, 1], repeat=len(measurement_indices) - len(postselected_indices)
            )
        ]

        measured_pauli_string = operator[0]
        measured_qubits = operator[1]

        if output is None:
            for i, q in enumerate(measured_qubits):
                if measured_pauli_string[i] == "X":
                    self.h(q)
                elif measured_pauli_string[i] == "Y":
                    self.s(q)
                    self.s(q)
                    self.s(q)
                    self.h(q)
            output = self.measure_all(measured_qubits, printing=printing)
            for i, q in enumerate(measured_qubits):
                if measured_pauli_string[i] == "X":
                    self.h(q)
                elif measured_pauli_string[i] == "Y":
                    self.h(q)
                    self.s(q)

        exp_vals = np.zeros(len(syndromes))

        error_coefficients = self.get_error_coefficients(errors_1q, errors_2q)

        thetas = np.zeros(self.n_magic) if thetas is None else np.array(thetas)
        theta_indices = self.theta_indices

        solution_phases = output.observable_result.phases
        solution_symbolic_phases = output.observable_result.symbolic_phases
        solution_error_terms = output.observable_result.error_terms
        solution_c_operators = output.observable_result.c_operators
        solution_s_operators = output.observable_result.s_operators

        solution_symbolic_phases = np.array(solution_symbolic_phases)
        surviving_solution_indices = np.arange(len(solution_phases))

        prod_phases = (
            np.ones(len(surviving_solution_indices), dtype=np.int8)
            - 2 * np.array(solution_phases)[surviving_solution_indices]
        )

        solution_symbolic_phases_filtered = solution_symbolic_phases[
            surviving_solution_indices, : self.n_m
        ]

        prod_error_terms = np.prod(
            error_coefficients[
                np.arange(len(solution_error_terms[0])),
                np.asarray(solution_error_terms)[surviving_solution_indices],
            ],
            axis=1,
        )

        prod_c_operators = np.prod(
            np.cos(thetas[theta_indices])[None, :]
            ** np.array(solution_c_operators)[surviving_solution_indices],
            axis=1,
        )
        prod_s_operators = np.prod(
            np.sin(thetas[theta_indices])[None, :]
            ** np.array(solution_s_operators)[surviving_solution_indices],
            axis=1,
        )

        solution_phases_trace = output.trace_result.phases
        solution_symbolic_phases_trace = output.trace_result.symbolic_phases
        solution_error_terms_trace = output.trace_result.error_terms
        solution_c_operators_trace = output.trace_result.c_operators
        solution_s_operators_trace = output.trace_result.s_operators

        solution_symbolic_phases_trace = np.array(solution_symbolic_phases_trace)
        surviving_solution_indices_trace = np.arange(len(solution_phases_trace))

        prod_phases_trace = (
            np.ones(len(surviving_solution_indices_trace), dtype=np.int8)
            - 2 * np.array(solution_phases_trace)[surviving_solution_indices_trace]
        )

        solution_symbolic_phases_filtered_trace = solution_symbolic_phases_trace[
            surviving_solution_indices_trace, : self.n_m
        ]

        prod_error_terms_trace = np.prod(
            error_coefficients[
                np.arange(len(solution_error_terms_trace[0])),
                np.asarray(solution_error_terms_trace)[
                    surviving_solution_indices_trace
                ],
            ],
            axis=1,
        )

        prod_c_operators_trace = np.prod(
            np.cos(thetas[theta_indices])[None, :]
            ** np.array(solution_c_operators_trace)[surviving_solution_indices_trace],
            axis=1,
        )
        prod_s_operators_trace = np.prod(
            np.sin(thetas[theta_indices])[None, :]
            ** np.array(solution_s_operators_trace)[surviving_solution_indices_trace],
            axis=1,
        )

        aggregate_prod = (
            prod_phases * prod_error_terms * prod_c_operators * prod_s_operators
        )
        aggregate_prod_trace = (
            prod_phases_trace
            * prod_error_terms_trace
            * prod_c_operators_trace
            * prod_s_operators_trace
        )

        batch_size = 1
        for i in range(0, len(syndromes), batch_size):
            batch_end = min(i + batch_size, len(syndromes))
            measurement_results = np.array(syndromes[i:batch_end])

            # Ensure measurement_results is always 2D
            if measurement_results.ndim == 1:
                measurement_results = measurement_results.reshape(1, -1)

            prod_symbolic_phases = 1 - 2 * (
                solution_symbolic_phases_filtered @ measurement_results.T % 2
            )

            exp_val = aggregate_prod @ prod_symbolic_phases

            prod_symbolic_phases_trace = 1 - 2 * (
                solution_symbolic_phases_filtered_trace @ measurement_results.T % 2
            )

            exp_val_trace = aggregate_prod_trace @ prod_symbolic_phases_trace

            # Ensure the result is always a 1D array for consistent assignment
            if exp_val.ndim == 0:
                exp_val = np.array([exp_val])
            if exp_val_trace.ndim == 0:
                exp_val_trace = np.array([exp_val_trace])

            exp_vals[i:batch_end] = exp_val / exp_val_trace

        for i in range(len(syndromes)):
            measurement_results = np.array(syndromes[i])

            # Ensure measurement_results is always 2D
            if measurement_results.ndim == 1:
                measurement_results = measurement_results.reshape(1, -1)

            prod_symbolic_phases = 1 - 2 * (
                solution_symbolic_phases_filtered @ measurement_results.T % 2
            )

            exp_val = aggregate_prod @ prod_symbolic_phases

            prod_symbolic_phases_trace = 1 - 2 * (
                solution_symbolic_phases_filtered_trace @ measurement_results.T % 2
            )

            exp_val_trace = aggregate_prod_trace @ prod_symbolic_phases_trace

            # Ensure the result is always a 1D array for consistent assignment
            if exp_val.ndim == 0:
                exp_val = np.array([exp_val])
            if exp_val_trace.ndim == 0:
                exp_val_trace = np.array([exp_val_trace])

            exp_vals[i] = exp_val / exp_val_trace

        exp_vals_dict = {
            syndrome[len(postselected_indices) :]: exp_val
            for syndrome, exp_val in zip(syndromes, exp_vals)
        }

        return exp_vals_dict
