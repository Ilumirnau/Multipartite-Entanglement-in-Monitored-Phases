import numpy as np
import sys
from z2_circuits import apply_row_S_xx, apply_row_NS_xx, C_matrices, sim_annealing

# DATA GENERATION


schedule = np.array([[3000, 1., np.pi/1.5], [3000, 0.8, np.pi/1.5],
                     [2500, 0.6, np.pi/1.5], [2000, 0.4, np.pi/1.5],
                     [2000, 0.2, np.pi/1.5], [1000, 0.1, np.pi/2],
                     [1000, 0.08, np.pi/2], [1000, 0.06, np.pi/2],
                     [1000, 0.04, np.pi/2], [1000, 0.02, np.pi/2],
                     [1000, 0.01, np.pi/4], [1000, 0.008, np.pi/4],
                     [1000, 0.004, np.pi/4], [1000, 0.002, np.pi/5],
                     ])
repeats = 5

L, seed, sym, invert = sys.argv[1:]
L, seed, sym, invert = int(L), int(seed), bool(eval(sym)), bool(eval(invert))

np.random.seed(seed * 13)
time = 4 * L

qfi_data = []
px = 0.0

for pu in np.linspace(0, 1-px, int((1-px)/0.05)+1):

    # print(f'pu: {pu}, px: {px}, L: {L}, seed: {seed}, invert: {invert}')
    state = np.zeros((2**L, 1), dtype=complex)  # |000...000>
    state[0] = 1

    for t in range(time):
        if sym:
            state = apply_row_S_xx(state, px, pu, L, invert)
        else:
            state = apply_row_NS_xx(state, px, pu, L, invert)

    C = C_matrices(state, L)
    C = 1/2 * np.real(C + np.transpose(C, axes=(0, 1, 3, 2)).conjugate())
    qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
    qfi_data.append([pu, qfi_traj, err_traj])

qfi_data = np.array(qfi_data)
if np.shape(qfi_data)[1] > 0:
    if sym:
        if invert:
            file_name = f'L{L}_seed{seed}_Z2i_xx_'
        else:
            file_name = f'L{L}_seed{seed}_Z2_xx_'
    else:
        if invert:
            file_name = f'L{L}_seed{seed}_Z2NSi_xx_'
        else:
            file_name = f'L{L}_seed{seed}_Z2NS_xx_'
    np.savetxt(f'/scratch/alirasol/z2_data/{file_name}.txt', qfi_data)
