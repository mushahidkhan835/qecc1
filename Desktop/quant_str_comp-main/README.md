# Quantum string comparison
This repo contains files related to comparing strings on a quantum computer. 
## Setup
To set up the environment, run
```bash
pip install qiskit pandas tabulate pylatexenc
```

## Usage examples
`string_comparison.py` contains the code for comparing a string against a set of strings.

`test_string_comparison.py` contains unit test cases for `string_comparison.py`. This file can also be interpreted as a 
set of examples of how `StringComparator` in `string_comparison.py` should be invoked. To run, do
```bash
python -m unittest test_string_comparison.py
```

For additional examples (containing invocations of the debug-related methods), run 
```bash
python string_comparison.py
```

## Get statistics for quantum circuits
The file compute_empirical_complexity.py simulates the generation of quantum circuits for string classification 
as described in the paper. Datasets found in `./datasets` (namely, Balance Scale, Breast Cancer, SPECT Heart, Tic-Tac-Toe Endgame, and Zoo) are taken from the UCI Machine Learning Repository.
To execute, run
```bash
python compute_empirical_complexity.py
```
The output is saved in `stats.csv` and `stats.json`.
