import numpy as np
import sys
from u1_circuits import C_matrices, sim_annealing

# DATA GENERATION


schedule = np.array([[2000, 1., np.pi/1.5], [2000, 0.8, np.pi/1.5],
                     [1500, 0.6, np.pi/1.5], [1500, 0.4, np.pi/1.5],
                     [2000, 0.2, np.pi/1.5], [1000, 0.1, np.pi/2],
                     [1000, 0.08, np.pi/2], [1000, 0.06, np.pi/2],
                     [1000, 0.04, np.pi/2], [1000, 0.02, np.pi/2],
                     [1000, 0.01, np.pi/4], [1000, 0.008, np.pi/4],
                     [1000, 0.004, np.pi/4], [1000, 0.002, np.pi/5],
                     ])
repeats = 5

L, seed, sym, invert = sys.argv[1:]
L, seed, sym, invert = int(L), int(seed), bool(eval(sym)), bool(eval(invert))

if sym:
    from u1_circuits import apply_row_S_z as apply_row
else:
    from u1_circuits import apply_row_NS_z as apply_row

np.random.seed(seed * 13)
time = 4 * L

qfi_data = []
pu = 0.0

for pz in np.linspace(0, 1-pu, int((1-pu)/0.05)+1):

    # print(f'pu: {pu}, px: {px}, L: {L}, seed: {seed}, invert: {invert}')
    state = 1/np.sqrt(2**L) * np.ones((2**L, 1), dtype=complex)  # |+++...+++>

    for t in range(time):
        state = apply_row(state, pz, pu, L, invert)

    C = C_matrices(state, L)
    C = 1/2 * np.real(C + np.transpose(C, axes=(0, 1, 3, 2)).conjugate())
    qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
    qfi_data.append([pu, qfi_traj, err_traj])

qfi_data = np.array(qfi_data)
if np.shape(qfi_data)[1] > 0:
    if sym:
        if invert:
            file_name = f'L{L}_seed{seed}_U1i_'
        else:
            file_name = f'L{L}_seed{seed}_U1_'
    else:
        if invert:
            file_name = f'L{L}_seed{seed}_U1NSi_'
        else:
            file_name = f'L{L}_seed{seed}_U1NS_'
    np.savetxt(f'/scratch/alirasol/u1_data/{file_name}.txt', qfi_data)
