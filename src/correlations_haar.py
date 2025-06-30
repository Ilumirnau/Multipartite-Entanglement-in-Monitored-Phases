# ------------------------------------------------------------------------------
"""
This script simulates the evolution of a quantum state under monitored Haar
random unitary operations and measurements.
It calculates the correlation matrices of the final state and saves
the results to a file.
Usage:
    python correlations_haar.py <L> <p> <seed>
Arguments:
    L (int): System size, the number of qubits.
    p (int): Measurement probability (in permille, i.e., parts per thousand).
    seed (int): Seed for the random number generator.
Description:
    The script performs the following steps:
    1. Parses the input arguments for system size (L),
       measurement probability (p), and random seed.
    2. Initializes the quantum state to |+++...+++>.
    3. Evolves the state over a number of time steps (4*L)
       by applying random measurements and unitary operations.
    4. Computes the correlation matrices of the final state.
    5. Reshapes the correlation matrices to a 2D array and saves them to
       a file in the 'data/haar_correlations/' directory.
    6. The filename format is 'L{L}_p{p}_t{time}_seed{seed}_C_'.

Example:
    python correlations_haar.py 10 500 42
"""

# ---------------------------------IMPORT LIBRARIES----------------------------
import sys

from monitored_haar import MeasureRow, UR, C_matrices
import numpy as np

# ----------------INPUT SYSTEM SIZE, MEASUREMENT PROBABILITY AND SEED----------

L, p, seed = sys.argv[1:]
L, p, seed = int(L), float(p), int(seed)
time = 4*L
np.random.seed(seed * 13)
p = p/1000
# ---------------------------------SIMULATION WITH THE GIVEN PARAMETERS--------

state = 1/np.sqrt(2**L) * np.ones((2**L, 1), dtype=complex)  # |+++...+++>

for t in range(time):
    state = MeasureRow(p, state, L)
    state = UR(state, t, L)

C = C_matrices(state, L)


# ---------------------------------SAVE DATA IN SUBFOLDER data/---------------

C2d = C.reshape((L**2, 9), order='C')  # reshaped matrix to 2d array
file_name = f'L{L}_p'+f'{p:.3f}'[2:]+f'_t{time}_seed{int(seed)}'
np.savetxt(f'data/haar_correlations/{file_name}_C_', C2d)
