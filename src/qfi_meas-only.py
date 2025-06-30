# ------------------------------------------------------------------------------

"""
This script calculates the Quantum Fisher Information (QFI) for a given system
size (L) and random seed.
The QFI is computed for different probabilities (ps) and saved to a file.

Usage:
    python qfi_meas-only.py <L> <seed>

Arguments:
    L    : System size (integer)
    seed : Random seed (integer)

Output:
    A file containing the QFI values for different probabilities,
    saved in the 'data/meas-only_qfi/' directory.
"""

# ------------------------------------------------------------------------------
import sys
import numpy as np

from measurement_only import layer, cc_qfi
# ------------------------------------------------------------------------------

L, seed = sys.argv[1:]
L, seed = int(L), int(seed)

np.random.seed(seed * 13)

t = 8 * L
ps = np.around(np.arange(0, 1.01, 0.01), decimals=3)
qfi = np.zeros((len(ps), 1))

for idx, p in enumerate(ps):
    S = np.zeros(L, dtype=int)
    for dt in range(t):
        S = layer(S, p)
    qfi[idx, 0] = cc_qfi(S)

file_name = f'L{L}_seed{int(seed)}'
np.savetxt(f'data/meas-only_qfi/{file_name}', qfi)
