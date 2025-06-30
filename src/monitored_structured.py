
# ------------------------------------------------------------------------------
"""This module provides functions for generating symmetry-preserving unitary
operators, applying triphase layers to quantum state simulators, and
calculating various entanglement measures and correlation matrices for
quantum states.
Functions:
    symmetry_preserving_us(size: int) -> list:
    triphase_layer(psi: stim.TableauSimulator, Us: list,
                    pz: float, pu: float) -> None:
    binaryMatrix(zStabilizers: list) -> np.ndarray:
        Construct the binary matrix representing the stabilizer states.
    getCutStabilizers(binaryMatrix: np.ndarray, cut) -> np.ndarray:
        Return only the part of the binary matrix that corresponds to the
        qubits we want to consider for a bipartition.
    gf2_rank(A: np.ndarray) -> int:
    ee(psi: stim.TableauSimulator, cut) -> int:
        Calculate the entanglement entropy of the circuit given a cut.
    S_topo(psi: stim.TableauSimulator) -> int:
        Returns the topological entanglement entropy.
    tmi(psi: stim.TableauSimulator) -> int:
        Calculate the tripartite mutual information of the system.
    corr_matrix(psi: stim.TableauSimulator) -> np.ndarray:
        Calculate the correlator matrix for the given quantum state simulator.
"""
import stim
import numpy as np
# ------------------------------------------------------------------------------


def symmetry_preserving_us(size: int) -> list:
    """
    Generate a list of symmetry-preserving unitary operators.
    This function generates all possible unsigned unitary operators of a given
    size and filters them to include only those that preserve the symmetry
    defined by the Pauli string "Z...Z".
    Args:
        size (int): The size of the unitary operators to generate.
    Returns:
        list: A list of symmetry-preserving unitary operators.
    """

    G = stim.PauliString("Z" * size).to_tableau()
    Us = []
    for u in stim.Tableau.iter_all(size, unsigned=True):
        if u*G*u.inverse() == G:
            Us.append(u)
    return Us


def triphase_layer(psi: stim.TableauSimulator, Us: list,
                   pz: float, pu: float) -> None:
    """
    Applies a triphase layer to the given quantum state simulator.
    This function performs three types of operations on the quantum state:
    1. Entangling measurements (XX) on pairs of qubits.
    2. Local measurements (Z) on individual qubits.
    3. Application of unitary operations from a given list.
    Parameters:
    psi (stim.TableauSimulator): The quantum state simulator to which the
                                 operations are applied.
    Us (list): A list of unitary operations to be applied.
    pz (float): The probability of applying a local Z measurement.
    pu (float): The probability of applying a unitary operation.
    Returns:
    None
    """

    N = psi.num_qubits
    probs = np.random.random(N)
    sites = np.random.permutation(range(N))

    # first we apply the entangling measurements XX
    xx = list(sites[(probs >= (pz+pu))])
    nxs = [[], [], []]

    for i in xx:
        if i not in nxs[0] and (i+1) % N not in nxs[0]:
            nxs[0].append(i)
            nxs[0].append((i+1) % N)
        elif i not in nxs[1] and (i+1) % N not in nxs[1]:
            nxs[1].append(i)
            nxs[1].append((i+1) % N)
        else:
            nxs[2].append(i)
            nxs[2].append((i+1) % N)

    ixs = [" ".join(map(str, nx)) for nx in nxs if nx]
    for ix in ixs:
        psi.do_circuit(stim.Circuit(f'MXX {ix}'))

    # then we apply the local measurements Z
    iz = ' '.join(map(str, list(sites[probs < pz])))
    psi.do_circuit(stim.Circuit(f'M {iz}'))

    # finally we apply the unitaries
    ius = sites[(probs >= pz) & (probs < (pz+pu))]
    us = np.random.choice(Us, len(ius))
    for site, u in zip(ius, us):
        psi.do_tableau(u, [site, (site+1) % N])


# ----------------------ENTANGLEMENT ENTROPY CALCULATION-----------------------
# all of them can be sped up with numba?
def binaryMatrix(zStabilizers: np.ndarray) -> np.ndarray:
    """
        - Purpose: Construct the binary matrix representing the stabilizer
        states.
        - Inputs:
            - zStabilizers (array): The result of conjugating the Z generators
              on the initial state.
        Outputs:
            - binaryMatrix (array of size (N, 2N)): An array that describes
              the location of the stabilizers in the tableau representation.
    """
    N = len(zStabilizers)
    binaryMatrix = np.zeros((N, 2*N), dtype=int)
    r = 0  # Row number
    for row in zStabilizers:
        c = 0  # Column number
        for i in row:
            if i == 3:  # Pauli Z
                binaryMatrix[r, N + c] = 1
            if i == 2:  # Pauli Y
                binaryMatrix[r, N + c] = 1
                binaryMatrix[r, c] = 1
            if i == 1:  # Pauli X
                binaryMatrix[r, c] = 1
            c += 1
        r += 1

    return binaryMatrix


def getCutStabilizers(binaryMatrix: np.ndarray, cut) -> np.ndarray:
    """
        - Purpose: Return only the part of the binary matrix that corresponds
          to the qubits we want to consider for a bipartition.
        - Inputs:
            - binaryMatrix (array of size (N, 2N)): The binary matrix for the
              stabilizer generators.
            - cut: location (int) or type (string) of cut
        - Outputs:
            - cutMatrix (array of size (N, 2cut)): The binary matrix for the
              cut on the left.
    """

    N = len(binaryMatrix)
    if isinstance(cut, int):
        cutMatrix = np.zeros((N, 2*cut))

        cutMatrix[:, :cut] = binaryMatrix[:, :cut]
        cutMatrix[:, cut:] = binaryMatrix[:, N:N+cut]

    elif isinstance(cut, str):
        if 'A' in cut:
            if 'B' in cut:
                subN = N//2
                cutMatrix = np.zeros((N, 2*subN))
                # x in ab
                cutMatrix[:, :subN] = binaryMatrix[:, :subN]
                # z in ab
                cutMatrix[:, subN:] = binaryMatrix[:, N:N+subN]

            elif 'C' in cut:
                subN = N//2
                cutMatrix = np.zeros((N, 2*subN))

                # x in a
                cutMatrix[:, :subN//2] = binaryMatrix[:, :subN//2]
                # z in a
                cutMatrix[:, subN:3*subN//2] = binaryMatrix[:, N:N+subN//2]
                # x in c
                cutMatrix[:, subN//2:subN] = binaryMatrix[:, subN:3*subN//2]
                # z in c
                cutMatrix[:, 3*subN//2:] = binaryMatrix[:, N+subN:N+3*subN//2]
            else:
                subN = N//4
                cutMatrix = np.zeros((N, 2*subN))
                # x in a
                cutMatrix[:, :subN] = binaryMatrix[:, :subN]
                # z in a
                cutMatrix[:, subN:] = binaryMatrix[:, N:N+subN]
        elif 'B' in cut:
            if 'C' in cut:
                subN = N//2
                cutMatrix = np.zeros((N, 2*subN))

                # x in bc
                cutMatrix[:, :subN] = binaryMatrix[:, subN//2:3*subN//2]
                # z in bc
                cutMatrix[:, subN:] = binaryMatrix[:, N+subN//2:N+3*subN//2]

            else:
                subN = N//4
                cutMatrix = np.zeros((N, 2*subN))
                # x in b
                cutMatrix[:, :subN] = binaryMatrix[:, subN:2*subN]
                # z in b
                cutMatrix[:, subN:] = binaryMatrix[:, N+subN:N+2*subN]
        elif 'C' in cut:
            subN = N//4
            cutMatrix = np.zeros((N, 2*subN))

            # x in c
            cutMatrix[:, :subN] = binaryMatrix[:, 2*subN:3*subN]
            # z in c
            cutMatrix[:, subN:] = binaryMatrix[:, N+2*subN:N+3*subN]

        else:
            subN = N//4
            cutMatrix = np.zeros((N, 2*subN))

            # x in d
            cutMatrix[:, :subN] = binaryMatrix[:, 3*subN:N]
            # z in d
            cutMatrix[:, subN:] = binaryMatrix[:, N+3*subN:]

    return cutMatrix


def gf2_rank(A: np.ndarray) -> int:
    """
    Find rank of a matrix over GF2.

    The rows of the matrix are given as nonnegative integers, thought
    of as bit-strings.

    This function modifies the input list. Use gf2_rank(rows.copy())
    instead of gf2_rank(rows) to avoid modifying rows.
    """
    matrix = np.mod(A, 2)

    # Get the shape of the matrix
    rows, cols = matrix.shape

    rank = 0
    for col in range(cols):
        # Find a row with a leading 1 in the current column
        for row in range(rank, rows):
            if matrix[row, col] == 1:
                # Swap the current row with the rank-th row
                matrix[[rank, row]] = matrix[[row, rank]]
                break
        else:
            # No leading 1 found, move to the next column
            continue

        # Eliminate the column below the leading 1
        for row in range(rank + 1, rows):
            if matrix[row, col] == 1:
                matrix[row] = (matrix[row] + matrix[rank]) % 2

        # Increment the rank
        rank += 1

    return rank


def ee(psi: stim.TableauSimulator, cut) -> int:
    '''
    This function calculates the entanglement entropy of the circuit given a
    cut.
    psi: Tableau state
    cut: number of qubits of the subsystem whose ent. entropy is being
    calculated (N_A) or type as string for TMI
    '''
    # Create the tableau representation
    tableau = psi.current_inverse_tableau() ** -1
    zs = [tableau.z_output(k) for k in range(len(tableau))]
    zs = np.array(zs)

    # Cut the binary matrix that corresponds to the qubits we want to consider
    # for a bipartition
    binMat = binaryMatrix(zs)
    cutMatrix = getCutStabilizers(binMat, cut)

    # Calculate the rank of the projected matrix via Gaussian elimination
    rank = gf2_rank(cutMatrix)

    N = psi.num_qubits
    # Calculate the entanglement entropy: S = rank(cutMatrix) - cut
    if isinstance(cut, int):
        cut_size = cut
    else:
        cut_size = N * len(cut) // 4

    S = rank - cut_size

    return S

# TOPOLOGICAL ENTANGLEMENT ENTROPY


def S_topo(psi: stim.TableauSimulator) -> int:
    '''
    Returns the topological entanglement entropy, the system's partition is
    ABDC (notice order)
    Stopo=Sab+Sbc-Sb-Sabc
    We name variables as in formula, but the function ee() is defined as ABCD,
    not ABDC
    '''
    sb = ee(psi, 'B')
    sab = ee(psi, 'AB')
    sbc = ee(psi, 'AC')
    sabc = ee(psi, 'C')

    return sab + sbc - sb - sabc


def tmi(psi: stim.TableauSimulator) -> int:
    '''

    Parameters
    ----------
    psi : stim.TableauSimulator

    Returns
    -------
    The tripartite mutual information of the system. The partition is ABCD
    I3 = Sa + Sb + Sc - Sab - Sac - Sbc + Sabc

    '''
    sa = ee(psi, 'A')
    sb = ee(psi, 'B')
    sc = ee(psi, 'C')
    sabc = ee(psi, 'D')

    sab = ee(psi, 'AB')
    sac = ee(psi, 'AC')
    sbc = ee(psi, 'BC')

    return sa + sb + sc + sabc - sab - sac - sbc
