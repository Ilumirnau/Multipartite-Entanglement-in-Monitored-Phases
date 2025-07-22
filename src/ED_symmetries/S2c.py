import numpy as np
import sys
from u1_circuits import apply_row_S, C_matrices, sim_annealing

# DATA GENERATION


schedule = np.array([[3500, 1., np.pi/1.5], [3500, 0.8, np.pi/1.5],
                     [3500, 0.6, np.pi/1.5], [3500, 0.4, np.pi/1.5],
                     [4000, 0.2, np.pi/1.5], [5000, 0.1, np.pi/2],
                     [5000, 0.08, np.pi/2], [5000, 0.06, np.pi/2],
                     [5000, 0.04, np.pi/2], [5000, 0.02, np.pi/2],
                     [3000, 0.01, np.pi/4], [3000, 0.008, np.pi/4],
                     [3000, 0.004, np.pi/4], [3000, 0.002, np.pi/5],
                     ])
repeats = 5

L, seed = sys.argv[1:]
L, seed = int(L), int(seed)

np.random.seed(seed * 13)
time = 4 * L

qfi_data = []
pu = 0.0
for pz in np.linspace(0, 1, 21):
    state = 1/np.sqrt(2**L) * np.ones((2**L, 1), dtype=complex)  # |+++...+++>

    for t in range(time):
        state = apply_row_S(state, pz, pu, L, False)

    C = C_matrices(state, L)
    C = 1/2 * np.real(C + np.transpose(C, axes=(0, 1, 3, 2)).conjugate())
    qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
    qfi_data.append([pz, qfi_traj, err_traj])

qfi_data = np.array(qfi_data)
if np.shape(qfi_data)[1] > 0:
    file_name = f'L{L}_seed{seed}_2_'
    np.savetxt(f'/scratch/alirasol/u1_data/{file_name}.txt', qfi_data)
