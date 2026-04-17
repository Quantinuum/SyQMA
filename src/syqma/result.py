"""Result containers returned by SyQMA measurements and evaluations."""

from math import isclose

import numpy as np

from .operations import Span


class MeasurementResult:
    """Store symbolic terms produced by a measurement reduction."""

    def __init__(
        self,
        phases: np.ndarray,
        symbolic_phases: np.ndarray,
        error_terms: np.ndarray,
        c_operators: np.ndarray,
        s_operators: np.ndarray,
        o_operators: np.ndarray,
        weights: np.ndarray,
        theta_indices: np.ndarray,
        list_error_channels: list[int],
        error_indices_list: list[np.ndarray],
    ) -> None:
        """Initialise the measurement result.

        Args:
            phases (numpy.ndarray): The phases of the measurement results.
            symbolic_phases (numpy.ndarray): The symbolic phases of the measurement results.
            error_terms (numpy.ndarray): The error terms of the measurement results.
            c_operators (numpy.ndarray): The c operators of the measurement results.
            s_operators (numpy.ndarray): The s operators of the measurement results.
            o_operators (numpy.ndarray): The o operators of the measurement results.
            weights (numpy.ndarray): The weights of the measurement results.
            theta_indices (numpy.ndarray): The theta indices of the measurement results.
            list_error_channels (list): The list of error channels.
            error_indices_list (list): The list of error indices.

        """
        # TODO convert everything to numpy arrays
        self.phases = phases
        self.symbolic_phases = symbolic_phases
        self.error_terms = error_terms
        self.c_operators = c_operators
        self.s_operators = s_operators
        self.o_operators = o_operators
        self.weights = weights
        self.theta_indices = theta_indices
        self.list_error_channels = list_error_channels
        self.error_indices_list = error_indices_list
        self.prod_phases = np.ones(len(phases), dtype=np.int8) - 2 * np.array(phases)
        self.prod_symbolic_phases = None
        self.prod_error_terms = None
        self.prod_c_operators = None
        self.prod_s_operators = None
        self.n_error_channels = len(error_terms)

    def set_measurement_results(self, measurement_results: list[int]) -> None:
        """Set the measurement-result signs used by symbolic terms."""
        self.prod_symbolic_phases = 1 - 2 * (
            self.symbolic_phases @ measurement_results % 2
        )

    def set_errors(
        self,
        errors_1q: float | list[float] | np.ndarray,
        errors_2q: float | list[float] | np.ndarray,
    ) -> None:
        """Set the Pauli-channel coefficients used by symbolic terms."""
        span = Span()
        span.list_error_channels = self.list_error_channels
        span.error_indices_list = self.error_indices_list
        error_coefficients = span.get_error_coefficients(errors_1q, errors_2q)
        self.prod_error_terms = np.prod(
            error_coefficients[
                np.arange(len(self.error_terms[0])), np.asarray(self.error_terms)
            ],
            axis=1,
        )

    def set_thetas(self, thetas: list[float]) -> None:
        """Set the trigonometric factors used by symbolic terms."""
        self.prod_c_operators = np.prod(
            np.cos(thetas[self.theta_indices])[None, :] ** self.c_operators, axis=1
        )
        self.prod_s_operators = np.prod(
            np.sin(thetas[self.theta_indices])[None, :] ** self.s_operators, axis=1
        )

    def print_symbolic_expression(
        self,
        errors_1q: list[float] | float = None,
        errors_2q: list[float] | float = None,
        measurement_results: list[int] = None,
        thetas: list[float] = None,
        sort_by: str = "n_errors",
    ) -> list[list[str]]:
        """Return LaTeX strings for the symbolic expression."""
        latex_string = ""
        span_error_ordering_1q = ["Z", "X", "Y"]
        span_error_ordering_2q = [
            "II",
            "IZ",
            "ZI",
            "ZZ",
            "IX",
            "IY",
            "ZX",
            "ZY",
            "XI",
            "XZ",
            "YI",
            "YZ",
            "XX",
            "XY",
            "YX",
            "YY",
        ]

        color_names = ["yellow", "cyan", "magenta", "red", "green", "pink", "orange"]
        colors = [color_names[i % len(color_names)] for i in range(len(self.phases))]

        terms = [[[], []], [[], []], []]
        all_solution_1q_n_errors = np.zeros(len(self.phases), dtype=int)
        all_solution_2q_n_errors = np.zeros(len(self.phases), dtype=int)

        if errors_1q is not None and errors_2q is not None:
            self.set_errors(errors_1q, errors_2q)
        if measurement_results is not None:
            self.set_measurement_results(measurement_results)
        if thetas is not None:
            thetas = np.array(thetas)
            self.set_thetas(thetas)

        term_prefactors = np.ones(len(self.phases), dtype=float)
        if self.prod_error_terms is not None:
            term_prefactors *= self.prod_error_terms
        if self.prod_c_operators is not None and self.prod_s_operators is not None:
            term_prefactors *= self.prod_c_operators * self.prod_s_operators
        if self.prod_symbolic_phases is not None:
            term_prefactors *= self.prod_symbolic_phases

        for i in range(len(self.phases)):
            latex_term = ""
            if measurement_results is None:
                exponent_string = ""
                for j in range(len(self.symbolic_phases[i])):
                    if self.symbolic_phases[i, j]:
                        if len(exponent_string) > 0:
                            exponent_string += "+"
                        exponent_string += rf"m_{{{j}}}"
                if len(exponent_string) > 0:
                    latex_term += rf"(-1)^{{{exponent_string}}}"
            if thetas is None:
                for j in range(self.c_operators.shape[1]):
                    if self.c_operators[i, j]:
                        latex_term += rf"\cos(\theta_{{{j}}})"
                    if self.s_operators[i, j]:
                        latex_term += rf"\sin(\theta_{{{j}}})"
                    if self.o_operators[i, j]:
                        latex_term = ""
                        break
            # bool_depolarising X bool_equal_gates
            latex_terms = [
                [latex_term, latex_term],
                [latex_term, latex_term],
                latex_term,
            ]
            if errors_1q is None:
                # TODO this assumes that the number of errors is the same for all gates (1q and 2q)
                gate_1q_n_errors = np.zeros(len(self.error_terms[i]), dtype=int)
                pauli_1q_n_errors = np.zeros(16, dtype=int)
                for j in range(len(self.error_terms[i])):
                    if self.error_terms[i, j] > 0 and self.list_error_channels[j] == 1:
                        pauli_1q_n_errors[self.error_terms[i, j]] += 1
                        gate_1q_n_errors[j] += 1
                        latex_terms[0][0] += (
                            rf"\lambda_{{1q,{j},{span_error_ordering_1q[self.error_terms[i, j]]}}}"
                        )
                for k in range(len(self.error_terms[i])):
                    if gate_1q_n_errors[k] > 1:
                        latex_terms[1][0] += (
                            rf"\lambda_{{1q,{k}}}^{{{gate_1q_n_errors[k]}}}"
                        )
                    elif gate_1q_n_errors[k] == 1:
                        latex_terms[1][0] += rf"\lambda_{{1q,{k}}}"
                for k in range(16):
                    if pauli_1q_n_errors[k] > 1:
                        latex_terms[0][1] += (
                            rf"\lambda_{{1q,{span_error_ordering_1q[k]}}}^{{{pauli_1q_n_errors[k]}}}"
                        )
                    elif pauli_1q_n_errors[k] == 1:
                        latex_terms[0][1] += (
                            rf"\lambda_{{1q,{span_error_ordering_2q[k]}}}"
                        )
                if np.any(gate_1q_n_errors):
                    all_solution_1q_n_errors[i] = np.sum(gate_1q_n_errors)
                    latex_terms[1][1] += (
                        rf"\lambda_{{1q}}^{{{all_solution_1q_n_errors[i]}}}"
                    )
                    latex_terms[2] += rf"(1-4p)^{{{all_solution_1q_n_errors[i]}}}"
            if errors_2q is None:
                # TODO this assumes that the number of errors is the same for all gates (1q and 2q)
                gate_2q_n_errors = np.zeros(len(self.error_terms[i]), dtype=int)
                pauli_2q_n_errors = np.zeros(16, dtype=int)
                for j in range(len(self.error_terms[i])):
                    if self.error_terms[i, j] > 0 and self.list_error_channels[j] == 2:
                        pauli_2q_n_errors[self.error_terms[i, j]] += 1
                        gate_2q_n_errors[j] += 1
                        latex_terms[0][0] += (
                            rf"\lambda_{{2q,{j},{span_error_ordering_2q[self.error_terms[i, j]]}}}"
                        )
                for k in range(len(self.error_terms[i])):
                    if gate_2q_n_errors[k] > 1:
                        latex_terms[1][0] += (
                            rf"\lambda_{{2q,{k}}}^{{{gate_2q_n_errors[k]}}}"
                        )
                    elif gate_2q_n_errors[k] == 1:
                        latex_terms[1][0] += rf"\lambda_{{2q,{k}}}"
                for k in range(16):
                    if pauli_2q_n_errors[k] > 1:
                        latex_terms[0][1] += (
                            rf"\lambda_{{2q,{span_error_ordering_2q[k]}}}^{{{pauli_2q_n_errors[k]}}}"
                        )
                    elif pauli_2q_n_errors[k] == 1:
                        latex_terms[0][1] += (
                            rf"\lambda_{{2q,{span_error_ordering_2q[k]}}}"
                        )
                if np.any(gate_2q_n_errors):
                    all_solution_2q_n_errors[i] = np.sum(gate_2q_n_errors)
                    latex_terms[1][1] += (
                        rf"\lambda_{{2q}}^{{{all_solution_2q_n_errors[i]}}}"
                    )
                    latex_terms[2] += rf"(1-16p)^{{{all_solution_2q_n_errors[i]}}}"

            for d in range(2):
                for g in range(2):
                    terms[d][g].append(latex_terms[d][g])
            terms[2].append(latex_terms[2])

        if sort_by == "n_measurements" and measurement_results is None:
            n_symbolic_phases = np.array(
                [self.symbolic_phases[i].size for i in range(len(self.phases))]
            )
            sorting_array = n_symbolic_phases
        elif sort_by == "n_errors" and errors_1q is None and errors_2q is None:
            sorting_array = all_solution_1q_n_errors + all_solution_2q_n_errors
        else:
            sorting_array = np.arange(len(self.phases))
        sorted_prefactors = term_prefactors[np.argsort(sorting_array)]

        latex_strings = [[], [], []]
        for d in range(2):
            for g in range(2):
                sorted_terms = [terms[d][g][i] for i in np.argsort(sorting_array)]
                grouped_terms = {}
                for i in range(len(self.phases)):
                    grouped_terms[sorted_terms[i]] = (
                        grouped_terms.get(sorted_terms[i], 0) + sorted_prefactors[i]
                    )
                sorted_terms = list(grouped_terms.keys())
                sorted_prefactors = list(grouped_terms.values())
                signs = [
                    "+" if prefactor > 0 else "-" for prefactor in sorted_prefactors
                ]
                prefactors = [
                    f"{abs(prefactor):.3f}" if not isclose(abs(prefactor), 1) else ""
                    for prefactor in sorted_prefactors
                ]
                for i in range(len(sorted_terms)):
                    if len(sorted_terms[i]) == 0:
                        sorted_terms[i] = "1"
                sorted_terms = [
                    rf"\textcolor{{{colors[i]}}}{{{signs[i] + ' ' + prefactors[i] + ' ' + term}}}"
                    for i, term in enumerate(sorted_terms)
                ]
                latex_string = (
                    rf"\left( {''.join(sorted_terms)} \right) / {len(self.phases)}"
                )
                latex_strings[d].append(latex_string)
        sorted_terms = [terms[2][i] for i in np.argsort(sorting_array)]
        grouped_terms = {}
        for i in range(len(self.phases)):
            grouped_terms[sorted_terms[i]] = (
                grouped_terms.get(sorted_terms[i], 0) + sorted_prefactors[i]
            )
        sorted_terms = list(grouped_terms.keys())
        sorted_prefactors = list(grouped_terms.values())
        signs = ["+" if prefactor > 0 else "-" for prefactor in sorted_prefactors]
        prefactors = [
            f"{abs(prefactor):.3f}" if not isclose(abs(prefactor), 1) else ""
            for prefactor in sorted_prefactors
        ]
        for i in range(len(sorted_terms)):
            if len(sorted_terms[i]) == 0:
                sorted_terms[i] = "1"
        sorted_terms = [
            rf"\textcolor{{{colors[i]}}}{{{signs[i] + ' ' + prefactors[i] + ' ' + term}}}"
            for i, term in enumerate(sorted_terms)
        ]
        latex_string = rf"\left( {''.join(sorted_terms)} \right) / {len(self.phases)}"
        latex_strings[2].append(latex_string)

        return latex_strings


class SimulationResult:
    """Store observable and trace measurement results."""

    def __init__(
        self,
        observable_result: MeasurementResult = None,
        trace_result: MeasurementResult = None,
        span: Span = None,
    ) -> None:
        """Initialise the simulation result.

        Args:
            observable_result (MeasurementResult): The observable result.
            trace_result (MeasurementResult): The trace result.
            span (Span): The span that produced the result.

        """
        self.observable_result = observable_result
        self.trace_result = trace_result
        self.span = span

    def save_to_file(self, filename: str) -> None:
        """Save the simulation result to a compressed .zstd file.

        Args:
            filename (str): The name of the file to which the simulator object will be saved.

        """
        import pickle
        import zstandard as zstd

        with zstd.open(
            f"{filename}.zstd", "wb", cctx=zstd.ZstdCompressor(threads=-1)
        ) as f:
            pickle.dump(self, f, protocol=-1)

    @classmethod
    def load_from_file(cls, filename: str) -> "SimulationResult":
        """Load a simulation result from a compressed .zstd file.

        Args:
            filename (str): The name of the file from which the simulator object will be loaded.

        """
        import pickle
        import zstandard as zstd

        with zstd.open(filename, "rb") as f:
            return pickle.load(f)

    def set_measurement_results(self, measurement_results: list[int]) -> None:
        """Set measurement-result signs on observable and trace results."""
        self.observable_result.set_measurement_results(measurement_results)
        self.trace_result.set_measurement_results(measurement_results)

    def set_errors(
        self,
        errors_1q: float | list[float] | np.ndarray,
        errors_2q: float | list[float] | np.ndarray,
    ) -> None:
        """Set Pauli-channel coefficients on observable and trace results."""
        self.observable_result.set_errors(errors_1q, errors_2q)
        self.trace_result.set_errors(errors_1q, errors_2q)

    def set_thetas(self, thetas: list[float]) -> None:
        """Set trigonometric factors on observable and trace results."""
        self.observable_result.set_thetas(thetas)
        self.trace_result.set_thetas(thetas)

    def print_symbolic_expression(self, *args, **kwargs) -> list[list[str]]:
        """Return LaTeX strings for the normalised symbolic expression."""
        latex_strings_observable = self.observable_result.print_symbolic_expression(
            *args, **kwargs
        )
        latex_strings_trace = self.trace_result.print_symbolic_expression(
            *args, **kwargs
        )

        latex_strings = [[], [], []]
        for d in range(2):
            for g in range(2):
                latex_string = rf"\frac{{{latex_strings_observable[d][g]}}}{{{latex_strings_trace[d][g]}}}"
                latex_strings[d].append(latex_string)
        latex_string = (
            rf"\frac{{{latex_strings_observable[2][0]}}}{{{latex_strings_trace[2][0]}}}"
        )
        latex_strings[2].append(latex_string)

        return latex_strings
