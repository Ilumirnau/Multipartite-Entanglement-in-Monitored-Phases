# ------------------------------------------------------------------------------
"""
This script performs simulated annealing on a set of matrices loaded from
files. The annealing schedule: number of iterations, temperature, and angle
variation parameter are defined in `schedule` array. The script reads matrices
from files, processes them, and applies simulated annealing respectively.
Arguments:
    L (int): The size parameter for the matrices.
    seed (int): The seed for random number generation.
Annealing Schedule:
    schedule (np.ndarray): A 2D array where each row defines the number of
                        iterations, temperature, and angle variation parameter
                        for a specific stage of the annealing process.
Constants:
    t (int): A constant derived from L, used in the filename pattern.
    repeats (int): The number of times the annealing process is repeated for
                   each matrix.
    skl (str): The filename pattern for loading matrices.
Variables:
    ann_qfi (list): List to store the QFI trajectory results from annealing.
    ann_err (list): List to store the last improvement value from annealing.
    p_seed (list): List to store the p values corresponding to each matrix.
Functions:
    sim_annealing(schedule, repeats, C, L): A function imported from the
                                            simulated_annealing module
                                            that performs the annealing process
                                            on a given matrix.
Process:
    1. Iterates over a range of p values.
    2. Constructs the filename for each p value.
    3. Checks if the file exists.
    4. Loads and processes the matrix from the file.
    5. Applies simulated annealing to the matrix.
    6. Stores the results in the corresponding lists.
    7. Saves the results to a file if any results are obtained.
"""

# ---------------------------------IMPORT LIBRARIES----------------------------
import os
import sys

import numpy as np
from simulated_annealing import sim_annealing

L, seed = sys.argv[1:]
L, seed = int(L), int(seed)

t = 4 * L

# annealing schedule:
# number of iterations at T, the temperature, delta (angle variation parameter)
schedule = np.array([[3500, 1., np.pi/1.5], [3500, 0.8, np.pi/1.5],
                     [3500, 0.6, np.pi/1.5], [3500, 0.4, np.pi/1.5],
                     [4000, 0.2, np.pi/1.5], [5000, 0.1, np.pi/2],
                     [5000, 0.08, np.pi/2], [5000, 0.06, np.pi/2],
                     [5000, 0.04, np.pi/2], [5000, 0.02, np.pi/2],
                     [3000, 0.01, np.pi/4], [3000, 0.008, np.pi/4],
                     [3000, 0.006, np.pi/4], [3000, 0.004, np.pi/4],
                     [3000, 0.002, np.pi/4],
                     ])
repeats = 5

ann_qfi = []
ann_err = []
p_seed = []

skl = 'data/haar_correlations/L{}_p{}_t{}_seed{}_C_'

for p in np.around(np.arange(0, 0.5, 0.01), decimals=3):
    pp = f'{p:1.3f}'[2:]
    fname = skl.format(L, pp, t, seed)
    if os.path.exists(fname):
        C = np.loadtxt(fname, dtype=complex).reshape(L, L, 3, 3)
        C = 1/2 * np.real(C + np.transpose(C, axes=(0, 1, 3, 2)).conjugate())
        qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
        ann_qfi.append(qfi_traj)
        ann_err.append(err_traj)
        p_seed.append(p)

QFI = np.array([ann_qfi, ann_err, p_seed])
if np.shape(QFI)[1] > 0:
    np.savetxt(f'data/haar_qfi/L{L}_seed{seed}_qfi_ann.txt', QFI)
