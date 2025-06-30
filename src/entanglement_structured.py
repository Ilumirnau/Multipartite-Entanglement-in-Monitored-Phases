# ------------------------------------------------------------------------------
"""
This script performs simulations of multipartite entanglement in monitored
phases using Clifford circuits.
The script takes two command-line arguments:
    L (int): The number of qubits.
    seed (int): The random seed for reproducibility.
The script performs the following steps:
1. Initializes the random seed.
2. Defines the number of time steps `t` as 4 times the number of qubits `L`.
3. Generates a list of (pz, pu) pairs where pz and pu are probabilities that
   sum to approximately 1.
4. For each (pz, pu) pair:
    a. Initializes a `stim.TableauSimulator` with the given seed and number of
       qubits `L`.
    b. Applies `triphase_layer` operations for `t` time steps.
    c. Computes the entanglement entropy (ee), topological entropy (S_topo),
       and total mutual information (`tmi`).
    d. Appends the results to `ent_data`.
5. Saves the entanglement data to a text file.
Imports:
    - numpy as np
    - stim
    - sys
    - symmetry_preserving_us, triphase_layer, ee, S_topo, tmi
      from monitored_clifford and monitored_structured modules
Functions:
    - symmetry_preserving_us: Generates symmetry-preserving unitaries.
    - triphase_layer: Applies a layer of triphase operations to the simulator.
    - ee: Computes the entanglement entropy.
    - S_topo: Computes the topological entropy.
    - tmi: Computes the total mutual information.
Outputs:
    - Entanglement data is saved in the 'data/structured_entanglement/'
      directory.
"""


# ------------------------------------------------------------------------------
import numpy as np
import stim
import sys

from monitored_structured import symmetry_preserving_us, triphase_layer
from monitored_structured import ee, S_topo, tmi
# ------------------------------------------------------------------------------

L, seed = sys.argv[1:]
L, seed = int(L), int(seed)
np.random.seed(seed * 13)
t = 4 * L

Us = symmetry_preserving_us(2)

ps = []
nps = np.linspace(0, 1, 51)
for pu in nps:
    for pz in nps:
        if ((1-pz-pu) <= 1.001 and (1-pz-pu) >= -0.001):
            ps.append((pz, pu))
ent_data = []
for pz, pu in ps:
    psi = stim.TableauSimulator(seed=seed)
    psi.set_num_qubits(L)

    for _ in range(t):
        triphase_layer(psi, Us, pz, pu)

    eent, stopo, tminfo = ee(psi, L//2), S_topo(psi), tmi(psi)
    ent_data.append([pz, pu, eent, stopo, tminfo])

np.savetxt(f'data/structured_entanglement/L{L}_seed{seed}_S_.txt',
           np.array(ent_data))
