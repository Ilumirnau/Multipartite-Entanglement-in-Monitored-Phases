# -------------------------------------------------------------------------------
"""
This script simulates a monitored Clifford circuit in brickwork structure
and computes the correlation matrix.
The script takes three command-line arguments:
    L (int): The number of qubits.
    prob (int): The probability (in permille) of a measurement occurring.
    seed (int): The random seed for reproducibility.
The script performs the following steps:
1. Initializes a TableauSimulator with the given seed.
2. Sets the number of qubits in the simulator to L.
3. Runs a loop for t = 4 * L iterations, where in each iteration:
    a. Applies a layer of measurements with probability p.
    b. Applies a layer of Clifford brickwork operations.
4. Computes the correlation matrix of the final state.
5. Saves the correlation matrix to a file in 'data/clifford_correlations/'.
Functions imported:
- layer_clifford_brickwork: Applies a layer of Clifford brickwork operations.
- layer_measurement: Applies a layer of measurements with a given probability.
- corr_matrix: Computes the correlation matrix of the current state.
The output file is named based on the input parameters.
"""

# ------------------------------------------------------------------------------
import numpy as np
import stim
import sys

from monitored_clifford import layer_clifford_brickwork, layer_measurement
from monitored_clifford import corr_matrix
# ------------------------------------------------------------------------------


L, prob, seed = sys.argv[1:]
L, prob, seed = int(L), int(prob), int(seed)
np.random.seed(seed * 13)
t = 4 * L
p = prob/1000

psi = stim.TableauSimulator(seed=seed)
psi.set_num_qubits(L)

for i in range(t):
    layer_measurement(psi, p)
    layer_clifford_brickwork(psi, i)

C = corr_matrix(psi)

file_name = f'L{L}_p'+f'{p:.3f}'[2:]+f'_seed{int(seed)}'
np.save(f'data/clifford_correlations/{file_name}_Cab_', C)
