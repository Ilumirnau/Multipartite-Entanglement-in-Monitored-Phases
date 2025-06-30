# ------------------------------------------------------------------------------
"""
This script calculates the Quantum Fisher Information (QFI) for a given system
size (L) and seed.
It uses a simulated annealing approach to optimize the QFI over a structured
set of parameters.
Functions:
- symmetry_preserving_us: Generates symmetry-preserving unitary operations.
- triphase_layer: Applies a triphase layer to the quantum state.
- corr_matrix: Computes the correlation matrix of the quantum state.
- sim_annealing: Performs simulated annealing to optimize the QFI.
Parameters:
- schedule (np.array): An array defining the annealing schedule with steps,
  temperature, and phase.
- repeats (int): Number of repetitions for the simulated annealing process.
- L (int): System size, passed as a command-line argument.
- seed (int): Random seed for reproducibility, passed as a command-line
  argument.
- t (int): Total number of time steps, calculated as 4 times the system size.
- Us (list): List of symmetry-preserving unitary operations.
- ps (list): List of tuples containing (pz, pu) values where pz and pu are
  probabilities.
- nps (np.array): Array of probabilities ranging from 0 to 1, divided into 26
  intervals.
- QFI (list): List to store the QFI results for different (pz, pu) pairs.
Usage:
Run the script from the command line with the system size (L) and seed as
arguments:
    python qfi_structured.py <L> <seed>
Output:
The QFI results are saved to a text file in the 'data/structured_qfi/surface/'
directory with the filename format 'L{L}_seed{seed}_qfi_ann.txt'.
"""

# ------------------------------------------------------------------------------
import numpy as np
import stim
import sys

from monitored_clifford import corr_matrix
from monitored_structured import symmetry_preserving_us, triphase_layer
from simulated_annealing import sim_annealing
# ------------------------------------------------------------------------------


schedule = np.array([[4000, 1., np.pi/1.5], [3500, 0.8, np.pi/1.5],
                     [3500, 0.6, np.pi/1.5], [3500, 0.4, np.pi/1.5],
                     [3500, 0.2, np.pi/1.5], [5000, 0.1, np.pi/2],
                     [5000, 0.08, np.pi/2], [5000, 0.06, np.pi/2],
                     [5000, 0.04, np.pi/2], [5000, 0.02, np.pi/4],
                     ])
repeats = 5


L, seed = sys.argv[1:]
L, seed = int(L), int(seed)
np.random.seed(seed * 13)
t = 4 * L

Us = symmetry_preserving_us(2)

ps = []
nps = np.linspace(0, 1, 26)
for pu in nps:
    for pz in nps:
        if ((1-pz-pu) <= 1.001 and (1-pz-pu) >= -0.001):
            ps.append((pz, pu))
QFI = []
for pz, pu in ps:
    psi = stim.TableauSimulator(seed=seed)
    psi.set_num_qubits(L)

    for _ in range(t):
        triphase_layer(psi, Us, pz, pu)

    C = corr_matrix(psi)
    qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
    QFI.append([pz, pu, qfi_traj, err_traj])
if np.shape(QFI)[1] > 0:
    np.savetxt(f'data/structured_qfi/surface/L{L}_seed{seed}_qfi_ann.txt', QFI)
