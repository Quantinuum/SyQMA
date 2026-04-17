"""Converters between external circuit formats and SyQMA simulators."""

import pytket
import stim
from .qec_simulator import QECSimulator


def circuit_from_pytket(circuit: pytket.Circuit, noise: bool = False) -> QECSimulator:
    """Convert a pytket circuit to a SyQMA simulator.

    Args:
        circuit (pytket.Circuit): The pytket circuit to convert.
        noise (bool, optional): Whether to apply Pauli noise after each gate. Defaults to False.

    Returns:
        QECSimulator: The circuit in the simulator representation.

    Raises:
        ValueError: If an unsupported operation is encountered.

    """
    sim = QECSimulator(n=circuit.n_qubits)

    for cmd in circuit.get_commands():
        op_type = cmd.op.type
        if op_type == pytket.OpType.H:
            sim.h(cmd.args[0].index[0])
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.X:
            sim.x(cmd.args[0].index[0])
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.Y:
            sim.y(cmd.args[0].index[0])
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.Z:
            sim.z(cmd.args[0].index[0])
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.S:
            sim.s(cmd.args[0].index[0])
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.Rz:
            sim.rz(cmd.args[0].index[0], sim.n_magic)
            if noise:
                sim.pauli_error_1(cmd.args[0].index[0])
        elif op_type == pytket.OpType.CX:
            sim.cnot(cmd.args[0].index[0], cmd.args[1].index[0])
            if noise:
                sim.pauli_error_2((cmd.args[0].index[0], cmd.args[1].index[0]))
        elif op_type == pytket.OpType.Measure:
            qubit = cmd.args[0].index[0]
            sim.measure(qubit)
        elif op_type == pytket.OpType.Reset:
            sim.reset(cmd.args[0].index[0])
        elif op_type == pytket.OpType.Barrier:
            pass
        else:
            raise ValueError(f"Unsupported operation: {op_type}")

    return sim


def circuit_from_stim_file(file_path: str, noise: bool = False) -> QECSimulator:
    """Convert a stim circuit file to a SyQMA simulator.

    Args:
        file_path (str): The path to the stim circuit file.
        noise (bool, optional): Whether to apply Pauli noise after each gate. Defaults to False.

    Returns:
        QECSimulator: The circuit in the simulator representation.

    """
    stim_circuit = stim.Circuit.from_file(file_path)

    sim = QECSimulator(n=stim_circuit.num_qubits, timing=True)

    # ! i qubit has not participated/been reset, might not have to add noise
    for cmd in stim_circuit:
        name = cmd.name
        targets = cmd.targets_copy()

        if name == "QUBIT_COORDS":
            continue
        elif name == "R":
            for target in targets:
                sim.reset(target.value)
        elif name == "RX":
            for target in targets:
                sim.reset(target.value, "X")
        elif name == "H":
            for target in targets:
                sim.h(target.value)
        elif name == "S":
            for target in targets:
                sim.s(target.value)
        elif name == "S_DAG":
            for target in targets:
                sim.sdag(target.value)
        elif name == "CX":
            for i_target in range(0, len(targets), 2):
                sim.cnot(targets[i_target].value, targets[i_target + 1].value)
        elif name == "M":
            for target in targets:
                sim.measure(target.value)
        elif name == "MX":
            for target in targets:
                sim.h(target.value)
                sim.measure(target.value)
                sim.h(target.value)
        elif name == "MPP":
            all_qubits = []
            all_paulis = []
            i = 0
            while i < len(targets):
                if targets[i].is_combiner:
                    all_qubits[-1].append(targets[i + 1].qubit_value)
                    all_paulis[-1] += targets[i + 1].pauli_type
                    i += 1
                else:
                    all_qubits.append([])
                    all_paulis.append("")
                    all_qubits[-1].append(targets[i].qubit_value)
                    all_paulis[-1] += targets[i].pauli_type
                i += 1
            for i in range(len(all_qubits)):
                sim.mpp(all_qubits[i], all_paulis[i])
        elif name == "DETECTOR":
            continue
        elif name == "OBSERVABLE_INCLUDE":
            continue
        elif name in ["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1"]:
            for target in targets:
                sim.pauli_error_1(target.value)
        elif name == "DEPOLARIZE2":
            for i_target in range(0, len(targets), 2):
                sim.pauli_error_2(
                    (targets[i_target].value, targets[i_target + 1].value)
                )
        elif name == "TICK":
            continue
        elif name == "SHIFT_COORDS":
            continue
        else:
            raise ValueError(f"Unsupported operation: {name}")

    return sim


def convert_latex_to_html(latex_string: str) -> str:
    """Convert a LaTeX string to an HTML document.

    Args:
        latex_string (str): The LaTeX string to convert.

    Returns:
        str: The HTML string.

    """
    html_string = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LaTeX Expression</title>
            <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.13.18/katex.min.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.13.18/katex.min.css">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                }}
                .latex-container {{
                    width: 100%;  /* Make the width dynamic */
                    text-align: center;
                    margin: 0 auto;
                    overflow-wrap: break-word;  /* Ensure text wraps instead of overflowing */
                }}
                .latex-container span {{
                    display: block;
                    white-space: normal;  /* Allow normal text wrapping */
                }}
            </style>
        </head>
        <body>
            <h2>LaTeX Expression with Colored Terms</h2>
            <p>
                <span id="latex-expr">{latex_string}</span>
            </p>

            <script type="text/javascript">
                var elem = document.getElementById('latex-expr');
                // Render in inline mode to allow wrapping
                katex.render(elem.textContent, elem, {{displayMode: false}});
            </script>
        </body>
        </html>
        """

    return html_string
