# SyQMA: Symbolic Quantum Memory-efficient Analyser

WARNING: This repository will be updated over the next few days.

The repository implements the strong simulation of universal quantum circuits exactly, analytically, and memory-efficiently, particularly tailored for quantum error correction.

## Installation
To be able to use the package, you need to install it. First, clone the repository. Then, assuming you already have a Python distribution, make sure `pip` is installed with:
```bash
python -m pip install --upgrade pip
```
Most users will be familiar with creating a new virtual environment, activating it and running the following command to install the package:
```bash
pip install -e .
```
Rather than using `pip`, we encourage the use of `uv` to install the package and manage the virtual environment. This is because `uv` is **much** faster and more user-friendly. It can be installed directly with `pip` as:
```bash
pip install uv
```
Next, automatically create a virtual environment in the `.venv` directory with the correct Python version and install the package and dependencies by simply running:
```bash
uv sync
```
Now, you can run any Python script in the repository with:
```bash
uv run my_script.py
```
To import the package in your Python script, use:
```python
import syqma
```

## Testing

To run the tests, simply use:
```bash
pytest
```
