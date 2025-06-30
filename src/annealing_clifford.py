# -------------------------------------------------------------------------------
"""
This script performs simulated annealing on Clifford correlations data to
compute the Quantum Fisher Information (QFI).
The script expects two command-line arguments:
1. L (int): System size.
2. seed (int): Random seed for reproducibility.
The script follows these steps:
1. Parses command-line arguments.
2. Defines the annealing schedule.
3. Initializes lists to store QFI and error trajectories.
4. Iterates over a range of probabilities `p` to load corresponding Clifford
   correlations data.
5. If the data file exists, it performs simulated annealing using the loaded
   data.
6. Stores the QFI and error trajectories.
7. Saves the results to a text file if any QFI data was computed.
Functions:
- sim_annealing(schedule, repeats, C, L): Performs simulated annealing on the
  given data.
File paths:
- Input: 'data/clifford_correlations/L{L}_p{pp}_seed{seed}_Cab_.npy'
- Output: 'data/clifford_qfi/L{L}_seed{seed}_qfi_ann.txt'
Modules required:
- numpy as np
- os
- sys
- simulated_annealing (custom module)
"""


# -------------------------------------------------------------------------------
import numpy as np
import os
import sys

from simulated_annealing import sim_annealing
# -------------------------------------------------------------------------------


L, seed = sys.argv[1:]
L, seed = int(L), int(seed)

t = 4 * L

schedule = np.array([[4000, 1., np.pi/1.5], [3500, 0.8, np.pi/1.5],
                     [3500, 0.6, np.pi/1.5], [3500, 0.4, np.pi/1.5],
                     [3500, 0.2, np.pi/1.5], [5000, 0.1, np.pi/2],
                     [5000, 0.08, np.pi/2], [5000, 0.06, np.pi/2],
                     [5000, 0.04, np.pi/2], [5000, 0.02, np.pi/2],
                     [500, 0.01, np.pi/4], [5000, 0.008, np.pi/4],
                     [5000, 0.006, np.pi/4], [5000, 0.004, np.pi/4],
                     [5000, 0.002, np.pi/4], [5000, 0.001, np.pi/6],
                     [5000, 0.0008, np.pi/6], [5000, 0.0006, np.pi/6],
                     [5000, 0.0004, np.pi/6], [5000, 0.0002, np.pi/6],
                     ])
repeats = 5

ann_qfi = []
ann_err = []

skl = 'data/clifford_correlations/L{}_p{}_seed{}_Cab_.npy'

for p in np.around(np.arange(0, 0.5, 0.01), decimals=3):
    pp = f'{p:1.3f}'[2:]
    fname = skl.format(L, pp, seed)
    if os.path.exists(fname):
        C = np.load(fname)
        qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
        ann_qfi.append(qfi_traj)
        ann_err.append(err_traj)
# first row qfi value, second row the last improvement when updating qfi
QFI = np.array([ann_qfi, ann_err])
if np.shape(QFI)[1] > 0:
    np.savetxt(f'data/clifford_qfi/L{L}_seed{seed}_qfi_ann.txt', QFI)
