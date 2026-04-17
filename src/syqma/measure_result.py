"""Measurement reduction routines for SyQMA simulators."""

import numpy as np
import time
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.traceback import install
from .result import MeasurementResult, SimulationResult
from .bitint import pack_rows_to_ints, pack_array_to_int, unpack_int_to_array
from .maths import solve_gf2_only
from .kernels import dfs_int
from .phase import _overall_phase_from_ops_mod4

install(width=1000, code_width=200, word_wrap=True)  # Enables pretty tracebacks
console = Console(log_time=False, log_path=False)


def _measure_all(
    simulator,
    measured_qubits: np.ndarray | list[int],
    printing: bool = True,
    trace: bool = False,
) -> SimulationResult:
    """Measure all qubits in the Z basis and calculate the expectation value analytically."""
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

    solution_phases = []
    solution_symbolic_phases = []
    solution_error_terms = []
    solution_c_operators = []
    solution_s_operators = []
    solution_o_operators = []
    solution_weights = []

    x0, K = solve_gf2_only(mat, vec)

    if x0.size == 0:
        return SimulationResult()

    d = K.shape[0]

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

    delta_c = K @ simulator.c_operators % 2
    delta_o = K @ simulator.o_operators % 2
    delta_c = delta_c.astype(np.uint8)
    delta_o = delta_o.astype(np.uint8)

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

    y = np.zeros(len(delta_c_int), dtype=np.uint8)
    valid = []

    if printing:
        console.print(
            Panel("[bold]🌲 Enumerating DFS Solutions...[/bold]", style="green")
        )
    dfs_int(0, c0_int, o0_int, y, valid, delta_c_int, delta_o_int, salv_int)

    print("Contributing Terms", f"{len(valid):,} / {2**d:,}")

    if simulator.timing:
        end_measure = time.time()
        if printing:
            console.log(
                f"[green]✓ DFS enumeration time:[/green] {end_measure - start_measure:.4f} s"
            )
        start_measure = time.time()

    time_cso = time_indices = time_einsum = 0
    time_pauli_sign = time_cso_sign = time_add_results = 0

    if printing:
        console.print(
            Panel(
                "[bold yellow]⚙️ Processing Valid DFS Solutions...[/bold yellow]",
                style="yellow",
            )
        )

    n_errors = len(simulator.error_indices_list)
    error_indices = np.zeros((simulator.n_stabs, n_errors))
    for i_error in range(n_errors):
        error_indices[: simulator.error_indices_list[i_error].size, i_error] = (
            simulator.error_indices_list[i_error]
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
        disable=not printing,
    ) as progress:
        if printing:
            task = progress.add_task(
                "[bold yellow]🚧 Processing DFS terms[/bold yellow]", total=len(valid)
            )

        for c_op, o_op, y in valid:
            start_cso = time.time() if simulator.timing else None

            solution_s_operator = c_op & o_op
            solution_c_operator = c_op ^ solution_s_operator
            solution_o_operator = o_op ^ solution_s_operator

            solution_c_operator = unpack_int_to_array(solution_c_operator, m)
            solution_s_operator = unpack_int_to_array(solution_s_operator, m)
            solution_o_operator = unpack_int_to_array(solution_o_operator, m)

            if simulator.timing:
                time_cso += time.time() - start_cso

            solution = (x0 + y @ K) % 2

            if simulator.timing:
                start_indices = time.time()
            solution_indices = np.nonzero(solution)[0]
            if simulator.timing:
                time_indices += time.time() - start_indices
                start_einsum = time.time()

            solution_phase = np.dot(solution, simulator.phases) % 2
            solution_symbolic_phase = solution @ simulator.symbolic_phases % 2

            selected_error_indices = error_indices[
                solution_indices, : len(simulator.error_indices_list)
            ]

            # assumes every location has either 1q or 2q errors (true)
            solution_error_term = np.bitwise_xor.reduce(
                selected_error_indices, axis=0, dtype=np.uint8
            )

            time_einsum += time.time() - start_einsum if simulator.timing else 0
            start_pauli_sign = time.time() if simulator.timing else None

            pauli_1_list = simulator.span_x[solution_indices]
            pauli_2_list = simulator.span_z[solution_indices]

            # Compute correct complex phase via mod-4 accumulation
            solution_phase_paulis = _overall_phase_from_ops_mod4(
                pauli_1_list, pauli_2_list
            )

            time_pauli_sign += time.time() - start_pauli_sign if simulator.timing else 0
            start_cso_sign = time.time() if simulator.timing else None

            c_op_list = simulator.c_operators[solution_indices]
            o_op_list = simulator.o_operators[solution_indices]

            solution_phase_cso = _overall_phase_from_ops_mod4(c_op_list, o_op_list)

            time_cso_sign += time.time() - start_cso_sign if simulator.timing else 0
            start_add_results = time.time() if simulator.timing else None

            solution_phase ^= np.uint8(
                (1 - np.real(solution_phase_paulis * solution_phase_cso)) // 2
            )

            solution_phases.append(solution_phase)
            solution_symbolic_phases.append(solution_symbolic_phase)
            solution_error_terms.append(solution_error_term)
            solution_c_operators.append(solution_c_operator)
            solution_s_operators.append(solution_s_operator)
            solution_o_operators.append(solution_o_operator)

            time_add_results += (
                time.time() - start_add_results if simulator.timing else 0
            )
            if printing:
                progress.advance(task)

    if simulator.timing:
        end_measure = time.time()
        if printing:
            console.log(
                f"[green]✓ Total evaluation time:[/green] {end_measure - start_measure:.4f} s"
            )
            console.log("[bold]Time breakdown:[/bold]")
            console.log(f"  [cyan]Indices:[/cyan] {time_indices:.4f} s")
            console.log(f"  [cyan]CSO Construction:[/cyan] {time_cso:.4f} s")
            console.log(
                f"  [cyan]Pauli Sign Computation:[/cyan] {time_pauli_sign:.4f} s"
            )

    # Summary Table
    if printing:
        table = Table(title="Stabiliser Simulation Summary", box=box.ROUNDED)
        table.add_column("Field", style="magenta bold")
        table.add_column("Value", style="green")

        table.add_row("Contributing Terms", f"{len(solution_phases):,} / {2**d:,}")

        console.print(table)

    observable_result = MeasurementResult(
        np.array(solution_phases),
        np.array(solution_symbolic_phases),
        np.array(solution_error_terms),
        np.array(solution_c_operators),
        np.array(solution_s_operators),
        np.array(solution_o_operators),
        np.array(solution_weights),
        theta_indices=simulator.theta_indices,
        list_error_channels=simulator.list_error_channels,
        error_indices_list=simulator.error_indices_list,
    )

    simulation_result = SimulationResult(observable_result=observable_result)

    if not trace:
        trace_result = _measure_all(
            simulator, measured_qubits=[], printing=printing, trace=True
        )
        simulation_result.trace_result = trace_result.observable_result

    simulation_result.span = simulator

    return simulation_result
