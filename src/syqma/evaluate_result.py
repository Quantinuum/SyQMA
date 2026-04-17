"""Expectation-value and marginal-probability evaluation routines."""

import os
import time
import numpy as np
from rich.traceback import install
from rich.panel import Panel
from rich.table import Table
from rich import box
from copy import deepcopy

from .result import SimulationResult
from rich.console import Console
from typing import Callable
import itertools

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from .bitint import (
    pack_rows_to_ints,
    pack_array_to_int,
    unpack_batch_of_ints_to_array_fast,
)
from .maths import solve_gf2_only
from .kernels import dfs_batch_generator
from .phase import _compute_batch_non_linear_phases

global_sum_order_0 = 0.0
global_sum_order_1 = 0.0
global_sum_order_2 = 0.0
global_sum_order_3 = 0.0
global_sum_order_4 = 0.0

install(width=1000, code_width=200, word_wrap=True)  # Enables pretty tracebacks
console = Console(log_time=False, log_path=False)


def time_function(func: Callable) -> Callable:
    """Print the runtime of the wrapped function."""

    def wrapper(*args, **kwargs) -> object:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\nTime taken by {func.__name__}: {end_time - start_time}s\n")
        return result

    return wrapper


def _evaluate_expectation_value(
    simulator,
    simulation_result: SimulationResult,
    errors_1q: float | list[float] = 0.0,
    errors_2q: float | list[float] = 0.0,
    measurement_results: list[int] = None,
    thetas: list[float] = None,
    normalise_by_trace: bool = True,
    weight_cutoff: int = None,
    theta_indices: list[int] = None,
    printing: bool = False,
) -> float:
    global \
        global_sum_order_0, \
        global_sum_order_1, \
        global_sum_order_2, \
        global_sum_order_3, \
        global_sum_order_4
    global dict_counts_main, prods_main, dict_counts_trace, prods_trace
    global counts_1q_main, counts_2q_main, counts_1q_trace, counts_2q_trace

    if simulation_result.observable_result is None:
        return 0.0

    solution_phases = simulation_result.observable_result.phases
    solution_symbolic_phases = simulation_result.observable_result.symbolic_phases
    solution_error_terms = simulation_result.observable_result.error_terms
    solution_c_operators = simulation_result.observable_result.c_operators
    solution_s_operators = simulation_result.observable_result.s_operators
    solution_weights = simulation_result.observable_result.weights

    if printing:
        console.print(
            Panel(
                "[bold white]📈 Evaluating Expectation Value[/bold white]", style="blue"
            )
        )

    if len(solution_phases) == 0:
        return 0.0

    start_all = time.time()

    time_cso = 0
    time_indices = 0
    time_einsum = 0
    time_pauli_sign = 0
    time_cso_sign = 0
    time_add_results = 0

    start_time = time.time()
    if isinstance(errors_1q, float):
        errors_1q = np.array([1 - 3 * errors_1q] + [errors_1q] * 3)
    if isinstance(errors_2q, float):
        errors_2q = np.array([1 - 15 * errors_2q] + [errors_2q] * 15)
    error_coefficients = simulator.get_error_coefficients(errors_1q, errors_2q)
    if printing:
        console.log(
            f"[green]✓ Error coefficients computed in {time.time() - start_time:.4f}s[/green]"
        )

    start_time = time.time()
    measurement_results_original = deepcopy(measurement_results)
    if measurement_results_original is None:
        measurement_results = np.full(simulator.n_m, np.nan)
    else:
        measurement_results = np.array(measurement_results)

    nan_mask = np.isnan(measurement_results)
    measurement_results_replaced_nan = measurement_results.copy()
    measurement_results_replaced_nan[nan_mask] = 0

    thetas = np.zeros(simulator.n_magic) if thetas is None else np.array(thetas)
    theta_indices = simulator.theta_indices if theta_indices is None else theta_indices
    if printing:
        console.log(
            f"[green]✓ Measurement inputs prepared in {time.time() - start_time:.4f}s[/green]"
        )

    start_time = time.time()
    solution_symbolic_phases = np.array(solution_symbolic_phases)
    surviving_solution_indices = np.nonzero(
        ~np.any(solution_symbolic_phases & nan_mask[None, :], axis=1)
    )[0]
    if printing:
        console.log(
            f"[green]✓ Symbolic phase filtering in {time.time() - start_time:.4f}s[/green]"
        )

    if printing:
        console.log(
            f"[cyan]Surviving terms:[/cyan] {len(surviving_solution_indices)} / {len(solution_phases)}"
        )

    contributing_terms = len(surviving_solution_indices)
    contributing_terms_cutoff = contributing_terms
    if weight_cutoff is not None:
        contributing_terms_cutoff = np.sum(
            np.array(solution_weights)[surviving_solution_indices] <= weight_cutoff
        )

    start_time = time.time()
    solution_symbolic_phases = solution_symbolic_phases[
        surviving_solution_indices, : simulator.n_m
    ]
    prod_symbolic_phases = 1 - 2 * (
        solution_symbolic_phases @ measurement_results_replaced_nan % 2
    )
    time_einsum += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ Symbolic phase product in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
    prod_error_terms = np.prod(
        error_coefficients[
            np.arange(len(solution_error_terms[0])),
            np.asarray(solution_error_terms)[surviving_solution_indices],
        ],
        axis=1,
    )
    time_einsum += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ Error term products in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
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
    time_einsum += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ c/s operator contributions in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
    prod_phases = (
        np.ones(len(surviving_solution_indices), dtype=np.int8)
        - 2 * np.array(solution_phases)[surviving_solution_indices]
    )
    time_indices += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ Solution phase indexing in {(time.time() - start_time):.4f}s[/green]"
        )

    error_array = np.asarray(solution_error_terms)[surviving_solution_indices]
    error_array_1q = error_array[
        :,
        [
            i
            for i in range(len(simulator.list_error_channels))
            if simulator.list_error_channels[i] == 1
        ],
    ]
    error_array_2q = error_array[
        :,
        [
            i
            for i in range(len(simulator.list_error_channels))
            if simulator.list_error_channels[i] == 2
        ],
    ]
    counts_1q = np.count_nonzero(error_array_1q, axis=1)
    counts_2q = np.count_nonzero(error_array_2q, axis=1)

    dict_counts = {
        (int(counts_1q[i]), int(counts_2q[i])): 0.0 for i in range(len(counts_1q))
    }
    for i in range(len(counts_1q)):
        dict_counts[(int(counts_1q[i]), int(counts_2q[i]))] += float(
            prod_phases[i]
            * prod_symbolic_phases[i]
            * prod_c_operators[i]
            * prod_s_operators[i]
        )

    if normalise_by_trace:
        dict_counts_main = dict_counts
        prods_main = (
            prod_phases * prod_symbolic_phases * prod_c_operators * prod_s_operators
        )
        counts_1q_main = counts_1q
        counts_2q_main = counts_2q
    else:
        dict_counts_trace = dict_counts
        prods_trace = (
            prod_phases * prod_symbolic_phases * prod_c_operators * prod_s_operators
        )
        counts_1q_trace = counts_1q
        counts_2q_trace = counts_2q

    start_time = time.time()
    exp_val = np.sum(
        prod_phases
        * prod_symbolic_phases
        * prod_error_terms
        * prod_c_operators
        * prod_s_operators
    )
    time_add_results += time.time() - start_time
    if printing:
        console.log(f"[green]✓ Final sum in {time_add_results:.4f}s[/green]")

    # Timing breakdown
    if printing:
        timing_table = Table(title="⏱ Timing Breakdown", box=box.ROUNDED)
        timing_table.add_column("Component", style="cyan")
        timing_table.add_column("Time (s)", justify="right", style="green")
        timing_table.add_row("CSO", f"{time_cso:.4f}")
        timing_table.add_row("Indices", f"{time_indices:.4f}")
        timing_table.add_row("Einsum", f"{time_einsum:.4f}")
        timing_table.add_row("Pauli Sign", f"{time_pauli_sign:.4f}")
        timing_table.add_row("CSO Sign", f"{time_cso_sign:.4f}")
        timing_table.add_row("Add Results", f"{time_add_results:.4f}")
        console.print(timing_table)

    if normalise_by_trace:
        start_time = time.time()
        exp_val_trace = _evaluate_expectation_value(
            simulator,
            SimulationResult(observable_result=simulation_result.trace_result),
            errors_1q,
            errors_2q,
            measurement_results=measurement_results,
            thetas=thetas,
            weight_cutoff=weight_cutoff,
            theta_indices=theta_indices,
            normalise_by_trace=False,
        )

        if exp_val_trace == 0:
            raise ValueError("State is unphysical, expectation value trace is 0")
        exp_val /= exp_val_trace
        if printing:
            console.print(f"[cyan]Normalized by trace: {exp_val_trace:.6f}[/cyan]")

    end_all = time.time()
    total_time = end_all - start_all
    if printing:
        console.log(
            f"[bold green]✓ Total evaluation time: {total_time:.4f}s[/bold green]"
        )

    # Summary table
    if printing:
        table = Table(title="🧾 Expectation Value Summary", box=box.ROUNDED)
        table.add_column("Metric", style="magenta bold")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Surviving Terms", f"{contributing_terms:,}")
        table.add_row("Used Terms (cutoff)", f"{contributing_terms_cutoff:,}")
        table.add_row("Expectation Value", f"{exp_val:.6f}")
        table.add_row("Total Time", f"{total_time:.2f} s")
        console.print(table)

    if weight_cutoff is not None and normalise_by_trace:
        return exp_val, contributing_terms, contributing_terms_cutoff
    return exp_val


def _evaluate_expectation_value_taylor(
    simulator,
    simulation_result: SimulationResult,
    measurement_results: list[int] = None,
    thetas: list[float] = None,
    normalise_by_trace: bool = True,
    weight_cutoff: int = None,
    theta_indices: list[int] = None,
    p: float = 1e-3,
    printing: bool = False,
) -> float:
    global \
        global_sum_order_0, \
        global_sum_order_1, \
        global_sum_order_2, \
        global_sum_order_3, \
        global_sum_order_4
    global dict_counts_main, prods_main, dict_counts_trace, prods_trace
    global counts_1q_main, counts_2q_main, counts_1q_trace, counts_2q_trace

    if simulation_result.observable_result is None:
        return 0.0

    solution_phases = simulation_result.observable_result.phases
    solution_symbolic_phases = simulation_result.observable_result.symbolic_phases
    solution_error_terms = simulation_result.observable_result.error_terms
    solution_c_operators = simulation_result.observable_result.c_operators
    solution_s_operators = simulation_result.observable_result.s_operators

    if printing:
        console.print(
            Panel(
                "[bold white]📈 Evaluating Expectation Value[/bold white]", style="blue"
            )
        )

    if len(solution_phases) == 0:
        return 0.0

    time_indices = 0
    time_einsum = 0

    start_time = time.time()
    measurement_results_original = deepcopy(measurement_results)
    if measurement_results_original is None:
        measurement_results = np.full(simulator.n_m, np.nan)
    else:
        measurement_results = np.array(measurement_results)

    nan_mask = np.isnan(measurement_results)
    measurement_results_replaced_nan = measurement_results.copy()
    measurement_results_replaced_nan[nan_mask] = 0

    thetas = np.zeros(simulator.n_magic) if thetas is None else np.array(thetas)
    theta_indices = simulator.theta_indices if theta_indices is None else theta_indices
    if printing:
        console.log(
            f"[green]✓ Measurement inputs prepared in {time.time() - start_time:.4f}s[/green]"
        )

    start_time = time.time()
    solution_symbolic_phases = np.array(solution_symbolic_phases)
    surviving_solution_indices = np.nonzero(
        ~np.any(solution_symbolic_phases & nan_mask[None, :], axis=1)
    )[0]
    if printing:
        console.log(
            f"[green]✓ Symbolic phase filtering in {time.time() - start_time:.4f}s[/green]"
        )

    if printing:
        console.log(
            f"[cyan]Surviving terms:[/cyan] {len(surviving_solution_indices)} / {len(solution_phases)}"
        )

    start_time = time.time()
    solution_symbolic_phases = solution_symbolic_phases[
        surviving_solution_indices, : simulator.n_m
    ]
    prod_symbolic_phases = 1 - 2 * (
        solution_symbolic_phases @ measurement_results_replaced_nan % 2
    )

    time_einsum += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ Symbolic phase product in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
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
    time_einsum += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ c/s operator contributions in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
    prod_phases = (
        np.ones(len(surviving_solution_indices), dtype=np.int8)
        - 2 * np.array(solution_phases)[surviving_solution_indices]
    )
    time_indices += time.time() - start_time
    if printing:
        console.log(
            f"[green]✓ Solution phase indexing in {(time.time() - start_time):.4f}s[/green]"
        )

    error_array = np.asarray(solution_error_terms)[surviving_solution_indices]
    error_array_1q = error_array[
        :,
        [
            i
            for i in range(len(simulator.list_error_channels))
            if simulator.list_error_channels[i] == 1
        ],
    ]
    error_array_2q = error_array[
        :,
        [
            i
            for i in range(len(simulator.list_error_channels))
            if simulator.list_error_channels[i] == 2
        ],
    ]
    counts_1q = np.count_nonzero(error_array_1q, axis=1)
    counts_2q = np.count_nonzero(error_array_2q, axis=1)

    dict_counts = {
        (int(counts_1q[i]), int(counts_2q[i])): 0.0 for i in range(len(counts_1q))
    }
    for i in range(len(counts_1q)):
        dict_counts[(int(counts_1q[i]), int(counts_2q[i]))] += float(
            prod_phases[i]
            * prod_symbolic_phases[i]
            * prod_c_operators[i]
            * prod_s_operators[i]
        )

    if normalise_by_trace:
        dict_counts_main = dict_counts
    else:
        dict_counts_trace = dict_counts

    if normalise_by_trace:
        _evaluate_expectation_value_taylor(
            simulator,
            SimulationResult(observable_result=simulation_result.trace_result),
            measurement_results=measurement_results,
            thetas=thetas,
            weight_cutoff=weight_cutoff,
            theta_indices=theta_indices,
            normalise_by_trace=False,
        )

    if normalise_by_trace:
        u = np.array(list(dict_counts_main.values()))
        v = np.array(list(dict_counts_trace.values()))

        a = np.array([k[0] for k in dict_counts_main.keys()])
        b = np.array([k[1] for k in dict_counts_main.keys()])
        c = np.array([k[0] for k in dict_counts_trace.keys()])
        d = np.array([k[1] for k in dict_counts_trace.keys()])

        start_time = time.time()
        taylor_coeffs(u, v, a, b, c, d, p=p, printing=True)
        time_taylor_coeffs = time.time() - start_time
        print()
        print(f"Time taylor coeffs: {time_taylor_coeffs}s")


def _process_batch_worker(
    chunk,
    K,
    x0,
    m,
    d,
    no_thetas,
    all_thetas,
    cos_tables,
    sin_tables,
    symbolic_phases_f32,
    phases_f32,
    err_bits0_f32,
    err_bits1_f32,
    err_bits2_f32,
    err_bits3_f32,
    error_coeffs,
    meas_mat_T_f32,
    post_nan_mask_u8,
    post_meas_replaced_u8,
    span_x,
    span_z,
    c_operators,
    o_operators,
) -> tuple[np.ndarray, np.ndarray, int]:
    current_batch_size = len(chunk[0])
    if current_batch_size == 0:
        return np.zeros(0), np.zeros(0), 0

    c_op, o_op, y_int_batch = chunk

    # Expand y_int to bits: (batch, d)
    shift = np.arange(d, dtype=np.uint64)
    y = ((y_int_batch[:, None] >> shift) & 1).astype(np.uint8, copy=False)

    # c_op and o_op are already numpy arrays
    solution_s_operator = c_op & o_op
    solution_c_operator = c_op ^ solution_s_operator

    if not no_thetas:
        solution_c_operators = unpack_batch_of_ints_to_array_fast(
            solution_c_operator, m
        )
        solution_s_operators = unpack_batch_of_ints_to_array_fast(
            solution_s_operator, m
        )

    solution = ((y @ K) & 1).astype(np.uint8, copy=False)
    solution ^= x0

    solution_f32 = solution.astype(np.float32, copy=False)

    solution_symbolic_phases = (solution_f32 @ symbolic_phases_f32).astype(
        np.uint8, copy=False
    ) & 1

    e0 = (solution_f32 @ err_bits0_f32).astype(np.uint8, copy=False) & 1
    e1 = (solution_f32 @ err_bits1_f32).astype(np.uint8, copy=False) & 1
    e2 = (solution_f32 @ err_bits2_f32).astype(np.uint8, copy=False) & 1
    e3 = (solution_f32 @ err_bits3_f32).astype(np.uint8, copy=False) & 1
    solution_error_terms = e0 | (e1 << 1) | (e2 << 2) | (e3 << 3)

    solution_phases = (solution_f32 @ phases_f32).astype(np.uint8, copy=False) & 1

    non_linear_corrections = _compute_batch_non_linear_phases(
        solution, span_x, span_z, c_operators, o_operators
    )
    solution_phases ^= non_linear_corrections

    prod_phases = 1 - 2 * solution_phases.astype(np.int8, copy=False)

    prod_error_terms = np.ones(
        (error_coeffs.shape[0], current_batch_size), dtype=error_coeffs.dtype
    )

    for j in range(solution_error_terms.shape[1]):
        idx = solution_error_terms[:, j]  # (batch,)
        prod_error_terms *= error_coeffs[:, j, :][:, idx]

    if no_thetas:
        theta_factor = np.ones((1, current_batch_size), dtype=float)
    else:
        c_mask = solution_c_operators.astype(bool, copy=False)
        s_mask = solution_s_operators.astype(bool, copy=False)
        theta_factor = np.empty((len(all_thetas), current_batch_size), dtype=float)
        for i_theta in range(len(all_thetas)):
            cos_sel = cos_tables[i_theta]
            sin_sel = sin_tables[i_theta]
            prod_c = np.prod(np.where(c_mask, cos_sel[None, :], 1.0), axis=1)
            prod_s = np.prod(np.where(s_mask, sin_sel[None, :], 1.0), axis=1)
            theta_factor[i_theta] = prod_c * prod_s

    parities = (
        solution_symbolic_phases.astype(np.float32, copy=False) @ meas_mat_T_f32
    ).astype(np.int8, copy=False) & 1
    prod_sym_T = (1 - 2 * parities).T

    base = (
        prod_phases[None, None, :]
        * prod_error_terms[:, None, :]
        * theta_factor[None, :, :]
    )
    base2 = base.reshape(-1, current_batch_size).T
    local_exp_vals_flat = prod_sym_T.astype(base2.dtype, copy=False) @ base2

    local_rescaled_prob = np.zeros((error_coeffs.shape[0], len(all_thetas)))

    survive_mask = ~np.any(solution_symbolic_phases & post_nan_mask_u8[None, :], axis=1)
    if np.any(survive_mask):
        prod_phases_f = prod_phases[survive_mask]
        sol_sym_f = solution_symbolic_phases[survive_mask]
        prod_error_terms_f = prod_error_terms[:, survive_mask]
        theta_factor_f = theta_factor[:, survive_mask]

        post_par = (sol_sym_f @ post_meas_replaced_u8) & 1
        post_sym = 1 - 2 * post_par.astype(np.int8, copy=False)

        local_rescaled_prob[:, :] += np.sum(
            prod_phases_f[None, None, :]
            * post_sym[None, None, :]
            * prod_error_terms_f[:, None, :]
            * theta_factor_f[None, :, :],
            axis=2,
        )

    return local_exp_vals_flat, local_rescaled_prob, current_batch_size


def _evaluate_expectation_values_from_scratch(
    simulator,
    measured_qubits: np.ndarray | list[int],
    all_errors_1q,
    all_errors_2q,
    all_measurement_results: list[list[int]] = None,
    all_thetas: list[list[float]] = None,
    postselected_indices: list[int] = [],
    weight_cutoff: int = None,
    batch_size: int = 1,
    theta_indices: list[int] = None,
    normalise_by_trace: bool = True,
    printing: bool = True,
    n_jobs: int = -1,
) -> float:
    start_all = time.time()

    if theta_indices is None:
        theta_indices = simulator.theta_indices

    if simulator.timing:
        start_measure = time.time()

    if isinstance(measured_qubits, list):
        measured_qubits = np.array(measured_qubits)

    alpha = simulator.span_x[:, : simulator.n_qubits].T
    a = np.zeros(simulator.n_qubits, dtype=np.uint8)

    non_measured_qubits = np.setdiff1d(np.arange(simulator.n_qubits), measured_qubits)

    if measured_qubits.size > 0:
        beta_Z = simulator.span_z[:, measured_qubits].T
        b_Z = np.ones(len(measured_qubits), dtype=np.uint8)

    beta_I = simulator.span_z[:, non_measured_qubits].T
    b_I = np.zeros(len(non_measured_qubits), dtype=np.uint8)

    if measured_qubits.size > 0:
        mat = np.vstack((alpha, beta_Z, beta_I))
        vec = np.hstack((a, b_Z, b_I))
    else:
        mat = np.vstack((alpha, beta_I))
        vec = np.hstack((a, b_I))

    x0, K = solve_gf2_only(mat, vec)
    # We rely heavily on fast parity arithmetic below
    x0 = np.asarray(x0, dtype=np.uint8)
    K = np.asarray(K, dtype=np.uint8)

    if x0.size == 0:
        if printing:
            console.print(
                "[yellow]System is inconsistent. Expectation value is 0.[/yellow]"
            )
        exp_vals = np.zeros(
            (len(all_measurement_results), len(all_errors_1q), len(all_thetas))
        )
        rescaled_prob = np.zeros((len(all_errors_1q), len(all_thetas)))
        probs = np.zeros_like(exp_vals)
        if normalise_by_trace is False:
            return exp_vals
        return exp_vals, probs

    x0_idx = np.nonzero(x0)[0]
    if x0_idx.size:
        c0 = np.bitwise_xor.reduce(
            simulator.c_operators[x0_idx], axis=0, dtype=np.uint8
        )
        o0 = np.bitwise_xor.reduce(
            simulator.o_operators[x0_idx], axis=0, dtype=np.uint8
        )
    else:
        c0 = np.zeros(simulator.n_magic, dtype=np.uint8)
        o0 = np.zeros(simulator.n_magic, dtype=np.uint8)

    d = K.shape[0]
    delta_c = (K @ simulator.c_operators % 2).astype(np.uint8)
    delta_o = (K @ simulator.o_operators % 2).astype(np.uint8)

    or_delta = delta_c | delta_o
    # Reorder rows to visit higher-impact branches first (fail-fast heuristic)
    if or_delta.shape[0] > 1:
        row_scores = np.sum(or_delta, axis=1)
        perm = np.argsort(-row_scores)
        K = K[perm]
        delta_c = delta_c[perm]
        delta_o = delta_o[perm]
        or_delta = or_delta[perm]
    salvageables = np.logical_or.accumulate(or_delta[::-1], axis=0)[::-1].astype(
        np.uint8
    )

    if simulator.timing:
        end_measure = time.time()
        if printing:
            console.log(
                f"[green]✓ DFS setup time:[/green] {end_measure - start_measure:.4f} s"
            )
        start_measure = time.time()

    m = c0.shape[0]

    c0_int = pack_array_to_int(c0)
    o0_int = pack_array_to_int(o0)
    delta_c_int = pack_rows_to_ints(delta_c)
    delta_o_int = pack_rows_to_ints(delta_o)
    salv_int = pack_rows_to_ints(salvageables)

    if simulator.timing:
        end_measure = time.time()
        if printing:
            console.log(
                f"[green]✓ Bit packing time:[/green] {end_measure - start_measure:.4f} s"
            )
        start_measure = time.time()

    y_int = 0

    if printing:
        console.print(
            Panel("[bold]🌲 Enumerating DFS Solutions...[/bold]", style="green")
        )
    valid_batches = dfs_batch_generator(
        c0_int, o0_int, y_int, delta_c_int, delta_o_int, salv_int, batch_size=batch_size
    )

    if simulator.timing:
        end_measure = time.time()
        if printing:
            console.log(
                f"[green]✓ DFS enumeration time:[/green] {end_measure - start_measure:.4f} s"
            )
        start_measure = time.time()

    simulator.time_cso = simulator.time_indices = simulator.time_einsum = 0
    simulator.time_pauli_sign = simulator.time_cso_sign = simulator.time_add_results = 0

    if printing:
        console.print(
            Panel(
                "[bold yellow]⚙️ Processing Valid DFS Solutions...[/bold yellow]",
                style="yellow",
            )
        )

    exp_vals = np.zeros(
        (len(all_measurement_results), len(all_errors_1q), len(all_thetas))
    )
    rescaled_prob = np.zeros((len(all_errors_1q), len(all_thetas)))

    all_error_coefficients = []
    for i_error in range(len(all_errors_1q)):
        errors_1q = all_errors_1q[i_error]
        errors_2q = all_errors_2q[i_error]

        error_coefficients = simulator.get_error_coefficients(errors_1q, errors_2q)
        all_error_coefficients.append(error_coefficients)

    no_thetas = len(all_thetas) == 1 and all_thetas[0] is None
    if not no_thetas:
        for i_theta, thetas in enumerate(all_thetas):
            all_thetas[i_theta] = (
                np.zeros(simulator.n_magic) if thetas is None else np.array(thetas)
            )

    n_errors = len(simulator.error_indices_list)
    error_indices = np.zeros((simulator.n_stabs, n_errors), dtype=np.uint8)
    for i_error in range(n_errors):
        error_indices[: simulator.error_indices_list[i_error].size, i_error] = (
            simulator.error_indices_list[i_error]
        )

    # Measurement results matrix (n_meas, n_m) as uint8
    meas_mat = np.asarray(all_measurement_results, dtype=np.uint8)
    if meas_mat.ndim == 1:
        meas_mat = meas_mat.reshape(1, -1)

    sym_dim = simulator.symbolic_phases.shape[1]
    if meas_mat.shape[1] != sym_dim:
        meas_mat = meas_mat[:, :sym_dim]

    symbolic_phases_f32 = np.asarray(simulator.symbolic_phases, dtype=np.float32)
    phases_f32 = np.asarray(simulator.phases, dtype=np.float32)

    # Postselection mask (kept identical to prior semantics)
    post_meas = np.array(
        [0] * len(postselected_indices)
        + [np.nan] * (simulator.n_m - len(postselected_indices)),
        dtype=float,
    )
    post_nan_mask = np.isnan(post_meas)
    post_meas_replaced = post_meas.copy()
    post_meas_replaced[post_nan_mask] = 0
    post_nan_mask_u8 = post_nan_mask[:sym_dim].astype(np.uint8, copy=False)
    post_meas_replaced_u8 = post_meas_replaced[:sym_dim].astype(np.uint8, copy=False)

    # Bit-sliced form of error indices for fast XOR-reduce
    # error_indices entries are Pauli indices in [0, 15]
    err_bits0_f32 = (error_indices & 1).astype(np.float32, copy=False)
    err_bits1_f32 = ((error_indices >> 1) & 1).astype(np.float32, copy=False)
    err_bits2_f32 = ((error_indices >> 2) & 1).astype(np.float32, copy=False)
    err_bits3_f32 = ((error_indices >> 3) & 1).astype(np.float32, copy=False)

    # Stack error coefficients so we can vectorize across all error configurations
    # shape: (n_error_configs, n_error_terms, 16)
    error_coeffs = np.stack(all_error_coefficients, axis=0)

    # Precompute trig tables per-theta (restricted to theta_indices)
    theta_indices_arr = np.asarray(theta_indices, dtype=int)
    cos_tables: list[np.ndarray] = []
    sin_tables: list[np.ndarray] = []
    if not no_thetas:
        for thetas in all_thetas:
            theta_sel = np.asarray(thetas, dtype=float)[theta_indices_arr]
            cos_tables.append(np.cos(theta_sel))
            sin_tables.append(np.sin(theta_sel))

    # OPTIMIZATION: Pre-transpose meas_mat for cache-friendly access
    meas_mat_T_f32 = np.ascontiguousarray(meas_mat.T).astype(
        np.float32, copy=False
    )  # (sym_dim, n_meas) - contiguous

    # OPTIMIZATION: Pre-flatten exp_vals for in-place accumulation
    exp_vals_flat = exp_vals.reshape(exp_vals.shape[0], -1)  # (n_meas, n_err*n_theta)

    # Process chunks in parallel using joblib
    from joblib import Parallel, delayed

    if printing:
        console.print(
            f"[cyan]Dispatching jobs across {n_jobs if n_jobs > 0 else os.cpu_count()} cores with batch size {batch_size} ...[/cyan]"
        )
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,  # <- keep progress after completion
        )
        progress.start()
        task = progress.add_task(
            "[bold yellow]🚧 Processing DFS terms[/bold yellow]", total=None
        )

    # Use a parallel loop since our generator now yields batches
    results = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
        delayed(_process_batch_worker)(
            batch,
            K,
            x0,
            m,
            d,
            no_thetas,
            all_thetas,
            cos_tables,
            sin_tables,
            symbolic_phases_f32,
            phases_f32,
            err_bits0_f32,
            err_bits1_f32,
            err_bits2_f32,
            err_bits3_f32,
            error_coeffs,
            meas_mat_T_f32,
            post_nan_mask_u8,
            post_meas_replaced_u8,
            simulator.span_x,
            simulator.span_z,
            simulator.c_operators,
            simulator.o_operators,
        )
        for batch in valid_batches
    )

    for local_exp_vals_flat, local_rescaled_prob, current_batch_size in results:
        exp_vals_flat[:] += local_exp_vals_flat
        rescaled_prob += local_rescaled_prob
        if printing:
            progress.advance(task, current_batch_size)

    if printing:
        progress.stop()

    rescaled_prob /= 2 ** len(postselected_indices)

    if printing:
        # Timing breakdown
        timing_table = Table(title="⏱ Timing Breakdown", box=box.ROUNDED)
        timing_table.add_column("Component", style="cyan")
        timing_table.add_column("Time (s)", justify="right", style="green")
        timing_table.add_row("Einsum", f"{simulator.time_einsum:.4f}")
        timing_table.add_row("Add Results", f"{simulator.time_add_results:.4f}")
        console.print(timing_table)

    if normalise_by_trace:
        exp_vals_trace = simulator.evaluate_expectation_values_from_scratch(
            [],
            all_errors_1q,
            all_errors_2q,
            all_measurement_results=all_measurement_results,
            all_thetas=all_thetas,
            postselected_indices=postselected_indices,
            weight_cutoff=weight_cutoff,
            theta_indices=theta_indices,
            normalise_by_trace=False,
            batch_size=batch_size,
        )
        if np.any(exp_vals_trace == 0):
            raise ValueError("State is unphysical, expectation value trace is 0")

        exp_vals /= exp_vals_trace

    end_all = time.time()
    total_time = end_all - start_all
    if printing:
        console.log(
            f"[bold green]✓ Total evaluation time: {total_time:.4f}s[/bold green]"
        )

    if printing:
        # Summary table
        table = Table(title="🧾 Expectation Value Summary", box=box.ROUNDED)
        table.add_column("Metric", style="magenta bold")
        table.add_column("Value", justify="right", style="green")
        table.add_row("Total Time", f"{total_time:.2f} s")
        console.print(table)

    if normalise_by_trace is False:
        return exp_vals

    probs = exp_vals_trace / (2 ** (simulator.n_m + d))
    return exp_vals, probs


def _get_marginal_probabilities(
    simulator,
    errors_1q,
    errors_2q,
    measurement_results: list[int] = None,
    thetas: list[float] = None,
) -> float:
    measurement_results_original = deepcopy(measurement_results)
    if measurement_results_original is None:
        measurement_results = np.full(simulator.n_m, np.nan)
    else:
        measurement_results = np.array(measurement_results)

    if simulator.outputs_probs is None:
        simulator.outputs_probs = [None] * simulator.n_m

    prob = 1

    for m in range(simulator.n_m):
        if np.isnan(measurement_results[m]):
            continue

        if simulator.outputs_probs[m] is None:
            simulator.outputs_probs[m] = simulator.measure_all(
                simulator.measurement_qubits[m]
            )

        output = simulator.outputs_probs[m]

        if len(output.observable_result.symbolic_phases[m]) == 0:
            prob *= 0.5
            continue

        measurement_results_m = np.zeros(simulator.n_m)
        measurement_results_m[:m] = measurement_results[:m]
        measurement_results_m[m:] = None

        exp_val = simulator.evaluate_expectation_value(
            output,
            errors_1q,
            errors_2q,
            measurement_results=measurement_results_m,
            thetas=thetas,
            normalise_by_trace=True,
        )

        current_prob = (1 + (1 - 2 * measurement_results[m]) * exp_val) / 2

        prob *= current_prob

    return prob


def _get_all_marginal_probabilities(
    simulator,
    errors_1q,
    errors_2q,
    thetas: list[float] = None,
    postselected_indices: list[int] = [],
    printing: bool = True,
) -> float:
    thetas = np.zeros(simulator.n_magic) if thetas is None else np.array(thetas)
    theta_indices = simulator.theta_indices

    probs = np.ones(2 ** (simulator.n_m - len(postselected_indices)))
    measurement_results_array = np.array(
        [
            (0,) * len(postselected_indices) + syndrome
            for syndrome in itertools.product(
                [0, 1], repeat=simulator.n_m - len(postselected_indices)
            )
        ],
        dtype=np.uint8,
    )

    if simulator.output_trace is None:
        simulator.output_trace = simulator.measure_all([])
    output = simulator.output_trace

    start_time = time.time()
    error_coefficients = simulator.get_error_coefficients(errors_1q, errors_2q)

    if printing:
        console.log(
            f"[green]✓ Error coefficients computed in {time.time() - start_time:.4f}s[/green]"
        )

    solution_phases = output.observable_result.phases
    solution_symbolic_phases = output.observable_result.symbolic_phases
    solution_error_terms = output.observable_result.error_terms
    solution_c_operators = output.observable_result.c_operators
    solution_s_operators = output.observable_result.s_operators

    start_time = time.time()
    solution_symbolic_phases = np.array(solution_symbolic_phases)
    surviving_solution_indices = np.arange(len(solution_phases))
    if printing:
        console.log(
            f"[green]✓ Symbolic phase filtering in {time.time() - start_time:.4f}s[/green]"
        )
        console.log(
            f"[cyan]Surviving terms:[/cyan] {len(surviving_solution_indices)} / {len(solution_phases)}"
        )

    contributing_terms = len(surviving_solution_indices)
    contributing_terms_cutoff = contributing_terms

    prod_phases = (
        np.ones(len(surviving_solution_indices), dtype=np.int8)
        - 2 * np.array(solution_phases)[surviving_solution_indices]
    )

    solution_symbolic_phases_filtered = solution_symbolic_phases[
        surviving_solution_indices, : simulator.n_m
    ]

    start_time = time.time()
    prod_error_terms = np.prod(
        error_coefficients[
            np.arange(len(solution_error_terms[0])),
            np.asarray(solution_error_terms)[surviving_solution_indices],
        ],
        axis=1,
    )
    if printing:
        console.log(
            f"[green]✓ Error term products in {(time.time() - start_time):.4f}s[/green]"
        )

    start_time = time.time()
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
    if printing:
        console.log(
            f"[green]✓ c/s operator contributions in {(time.time() - start_time):.4f}s[/green]"
        )

    aggregate_prod = (
        prod_phases * prod_error_terms * prod_c_operators * prod_s_operators
    )

    batch_size = 8
    if printing:
        print(f"batch_size = {batch_size}")
    for i in range(0, len(measurement_results_array), batch_size):
        if printing:
            console.print(
                Panel(
                    "[bold white]📈 Evaluating Expectation Value[/bold white]",
                    style="blue",
                )
            )
        start_all = time.time()

        simulator.time_einsum = 0
        simulator.time_add_results = 0

        batch_end = min(i + batch_size, len(measurement_results_array))
        measurement_results = measurement_results_array[i:batch_end]

        # Ensure measurement_results is always 2D
        if measurement_results.ndim == 1:
            measurement_results = measurement_results.reshape(1, -1)

        start_time = time.time()
        prod_symbolic_phases = 1 - 2 * (
            (solution_symbolic_phases_filtered @ measurement_results.T) % 2
        )
        simulator.time_einsum += time.time() - start_time
        if printing:
            console.log(
                f"[green]✓ Symbolic phase product in {(time.time() - start_time):.4f}s[/green]"
            )

        start_time = time.time()
        exp_vals = aggregate_prod @ prod_symbolic_phases
        simulator.time_add_results += time.time() - start_time
        if printing:
            console.log(
                f"[green]✓ Final sum in {simulator.time_add_results:.4f}s[/green]"
            )

        # Ensure exp_vals is always 1D for consistent assignment
        if exp_vals.ndim == 0:
            exp_vals = np.array([exp_vals])
        elif exp_vals.ndim > 1:
            exp_vals = exp_vals.flatten()

        # Timing breakdown
        if printing:
            timing_table = Table(title="⏱ Timing Breakdown", box=box.ROUNDED)
            timing_table.add_column("Component", style="cyan")
            timing_table.add_column("Time (s)", justify="right", style="green")
            timing_table.add_row("Einsum", f"{simulator.time_einsum:.4f}")
            timing_table.add_row("Add Results", f"{simulator.time_add_results:.4f}")
            console.print(timing_table)

        end_all = time.time()
        total_time = end_all - start_all
        if printing:
            console.log(
                f"[bold green]✓ Total evaluation time: {total_time:.4f}s[/bold green]"
            )

        # Summary table
        if printing:
            table = Table(title="🧾 Expectation Value Summary", box=box.ROUNDED)
            table.add_column("Metric", style="magenta bold")
            table.add_column("Value", justify="right", style="green")
            table.add_row("Surviving Terms", f"{contributing_terms:,}")
            table.add_row("Used Terms (cutoff)", f"{contributing_terms_cutoff:,}")
            table.add_row("Total Time", f"{total_time:.2f} s")
            console.print(table)

        exp_vals /= 2**simulator.n_m

        probs[i:batch_end] = exp_vals

    rescaled_prob = simulator.evaluate_expectation_value(
        output,
        errors_1q,
        errors_2q,
        measurement_results=[0] * len(postselected_indices)
        + [np.nan] * (simulator.n_m - len(postselected_indices)),
        thetas=thetas,
        normalise_by_trace=False,
    )
    rescaled_prob /= 2 ** len(postselected_indices)

    probs /= rescaled_prob

    prob_dict = {}

    for idx, measurement_results in enumerate(measurement_results_array):
        prob_dict[tuple(map(int, measurement_results[len(postselected_indices) :]))] = (
            float(probs[idx])
        )

    return prob_dict


@time_function
def taylor_coeffs(
    u: np.ndarray,
    v: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    max_order: int = 6,
    p: float = 1e-3,
    printing: bool = False,
) -> None:
    """Compute Taylor coefficients A_n.

    f(p) = (Σ_i u_i (1 - 4p/3)^{a_i} (1 - 16p/15)^{b_i}) /
           (Σ_j v_j (1 - 4p/3)^{c_j} (1 - 16p/15)^{d_j})
    up to order max_order (inclusive).
    """
    n_terms = max_order + 1

    # α_i,n for numerator; β_j,n for denominator
    alpha = np.zeros((len(u), n_terms))
    beta = np.zeros((len(v), n_terms))

    alpha[:, 0] = 1.0
    beta[:, 0] = 1.0

    # --- Recursively compute α_i,n and β_j,n ---
    for n in range(max_order):
        for i in range(len(u)):
            alpha[i, n + 1] = -(
                4 / 3 * a[i] * np.sum(4 ** np.arange(n + 1) * alpha[i, n::-1])
                + 16 / 15 * b[i] * np.sum(16 ** np.arange(n + 1) * alpha[i, n::-1])
            ) / (n + 1)
        for j in range(len(v)):
            beta[j, n + 1] = -(
                4 / 3 * c[j] * np.sum(4 ** np.arange(n + 1) * beta[j, n::-1])
                + 16 / 15 * d[j] * np.sum(16 ** np.arange(n + 1) * beta[j, n::-1])
            ) / (n + 1)

    # --- Aggregate into N_n and D_n ---
    N = np.dot(u, alpha)
    D = np.dot(v, beta)

    # --- Compute A_n recursively ---
    A = np.zeros(n_terms)
    A[0] = N[0] / D[0]
    if printing:
        print(f"A[0]: {N[0]} {D[0]} {A[0]}")
    for n in range(1, n_terms):
        A[n] = (N[n] - np.dot(D[1 : n + 1], A[n - 1 :: -1])) / D[0]

    if printing:
        print()
        for n in range(1, n_terms):
            print(f"A[{n}] p^{n}: {A[n]} {A[n] * p**n}")
        print()

    EV = 0.0

    for n in range(n_terms):
        EV += A[n] * p**n
        if printing:
            print(f"EV{n}: {EV}")

    print()
    for n in range(n_terms):
        if printing:
            if n > 0 and A[n] != 0:
                print(f"LER[{n}]: {-A[n] * p**n / 2}")
                print(f"LER_coefficient[{n}]: {-A[n] / 2}")
