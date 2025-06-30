# ------------------------------------------------------------------------------
"""
This module contains functions to simulate monitored random quantum circuits
using Haar random gates.
Functions:
    random_unitary(size):
        Generates a Haar random unitary matrix of dims. 2^size x 2^size.
    is1(index, d):
        Checks if there is a 1 in the binary representation of integer d at
        the given index.
    bit_assign(tuple_lst, N):
        Assigns bits in tuple_lst positions to bit string integer N.
    U_ij(index1, index2, state, size):
        Applies a random unitary gate on the state between the given indices.
    UR(state, time, size):
        Applies random unitary gates on the state for a given time and size.
    measure(p, index, state, size):
        Performs a projective measurement of the state on the spin z direction
        with probability p.
    MeasureRow(p, state, size):
        Measures the state on all sites with probability p.
    C_matrices(state, size):
        Computes the correlation matrices for the given state and system size.
Usage:
    The script can be run with the following command line arguments:
        L: System size
        p: Measurement probability
        seed: Random seed for reproducibility
"""

# ---------------------------------IMPORT LIBRARIES----------------------------
import numpy as np
from scipy import stats

# ---------------------------------DEFINE NECESSARY FUNCTIONS------------------


def random_unitary(size: int) -> np.ndarray:
    """
    Generate a random unitary matrix of dimensions 2^size x 2^size.
    Parameters:
    size (int): Amount of qubits on which the gate is applied. The
    resulting matrix will be of dimensions 2^size x 2^size.
    Returns:
    numpy.ndarray: A random unitary matrix of dimensions 2^size x 2^size.
    """

    h = stats.unitary_group.rvs(2**size)
    return h


def is1(index: int, d: int) -> int:
    """
    Check if the bit at a specific index is set to 1 in the binary form of d.
    Args:
        index (int): The position of the bit to check (0-based index).
        d (int): The integer whose binary representation is to be checked.
    Returns:
        int: A non-zero integer if the bit at the specified index is 1, else 0.
    """

    return d & (1 << index)


def bit_assign(tuple_lst: list, N: int) -> int:
    """
    Assigns bits to the integer N based on the provided list of tuples.
    Parameters:
    tuple_lst (list of tuples): A list of tuples where each tuple contains an
    index and a value (idx, val).
            The index represents the bit position, and the value represents
            the bit value (0 or 1).
    N (int): The integer to which the bits will be assigned.
    Returns:
    int: The new integer value after the bits have been assigned.
    Example:
    >>> bit_assign([(2, 1)], 5)
    13
    >>> bit_assign([(1, 0), (3, 1)], 8)
    40
    """

    if len(tuple_lst) == 1:
        (idx0, val0) = tuple_lst[0]
        i = val0 << idx0
        Nbar = N & ((1 << idx0) - 1)
        N1 = ((N - Nbar) << 1) + i + Nbar

        return N1
    else:
        t = sorted(tuple_lst)
        (idx0, val0), (idx1, val1) = t
        i = val0 << idx0
        j = val1 << idx1

        Nbar = N & ((1 << idx0) - 1)
        N1 = ((N - Nbar) << 1) + i + Nbar

        Nbar = N1 & ((1 << idx1) - 1)
        N2 = ((N1 - Nbar) << 1) + j + Nbar
        return N2


def U_ij(index1: int, index2: int,
         state: np.ndarray, size: int) -> np.ndarray:
    """
    Applies a random 2x2 unitary transformation to the specified qubits in the
    given quantum state.
    Parameters:
    index1 (int): The index of the first qubit.
    index2 (int): The index of the second qubit.
    state (numpy.ndarray): The current state vector of the quantum system.
    size (int): The number of qubits in the system.
    Returns:
    numpy.ndarray: The new state vector after applying the unitary gate.
    """

    new_state = np.zeros_like(state)
    hg = random_unitary(2)
    for z in range(2**(size-2)):
        row00 = bit_assign(((index1, 0), (index2, 0)), z)
        row01 = bit_assign(((index1, 0), (index2, 1)), z)
        row10 = bit_assign(((index1, 1), (index2, 0)), z)
        row11 = bit_assign(((index1, 1), (index2, 1)), z)

        new_state[[row00, row01, row10, row11]] = hg @ state[[row00, row01,
                                                              row10, row11]]
    return new_state


def UR(state: np.ndarray, time: int, size: int) -> np.ndarray:
    """
    Applies a unitary operation to the given state in a specific pattern.
    This function iterates over the state in steps of 2, starting from 0 or 1
    depending on the parity of the given time. For each iteration, it applies
    the unitary operation U_ij to pairs of adjacent elements in the state.
    Parameters:
    state (any): The initial state to which the unitary operation is applied.
    time (int): The current time step, used to determine the starting index.
    size (int): The system size.
    Returns:
    any: The state after applying the unitary operations.
    """

    for i in range(time % 2, size, 2):
        state = U_ij(i, (i+1) % size, state, size)
    return state


def measure(p: float, index: int, state: np.ndarray, size: int) -> np.ndarray:
    """
    Measures a state with a given probability and updates the state
    accordingly.
    Parameters:
    p (float): The probability of performing the measurement.
    index (int): The index of the qubit to be measured.
    state (numpy.ndarray): The quantum state vector.
    size (int): The number of qubits in the system.
    Returns:
    numpy.ndarray: The updated quantum state vector after measurement.
    """

    if np.random.rand() < p:
        # measure
        pUp = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                     if not is1(index, i)])
        # check result of measurement, up (0) with p pUp
        if np.random.rand() < pUp:
            for c in range(2**size):
                if is1(index, c):
                    state[c] = 0
            new_state = state/np.sqrt(pUp)
        else:
            for c in range(2**size):
                if not is1(index, c):
                    state[c] = 0
            new_state = state/np.sqrt(1-pUp)
        return new_state
    else:
        # do nothing to state
        return state


def MeasureRow(p: float, state: np.ndarray, size: int) -> np.ndarray:
    """
    Measures each qubit of the given quantum state with a certain probability.
    Parameters:
    p (float): The probability of measurement.
    state (np.ndarray): The quantum state represented as a numpy array.
    size (int): The size of the system (number of rows to measure).
    Returns:
    np.ndarray: The quantum state after measurements.
    """

    for i in range(size):
        state = measure(p, i, state, size)
    return state


def C_matrices(state: np.ndarray, size: int) -> np.ndarray:
    """
    Calculate the connected correlation matrices for a given quantum state.
    Parameters:
    state (numpy.ndarray): The quantum state vector of size 2**size.
    size (int): The size of the system (number of sites).
    Returns:
    numpy.ndarray: A 4-dimensional array of shape (size, size, 3, 3)
                   containing the correlation matrices.
                   The first two dimensions correspond to the site indices,
                   and the last two dimensions correspond
                   to the correlation directions (x, y, z).
    Notes:
    - The function computes single-site and two-site expectation values for
      the Pauli matrices (x, y, z).
    - The correlation matrices are calculated by subtracting the product of
      single-site expectation values from the two-site expectation values.
    """
    # single site expectation values (x,y,z)
    evs = np.zeros((size, 3), dtype=np.complex128)
    # two-site expectation values (xx,xy,xz,yx,yy,yz,zx,zy,zz), left to right
    evs2 = np.zeros((size, size, 3, 3), dtype=np.complex128)

    for site1 in range(size):
        for k in range(2**size):
            # state k-th component
            s_k = state[k].item()
            # k with site1 bit flipped
            k_f1 = k ^ (1 << site1)
            # 1 or -1 depending on whether k has a 0 or 1 on site1
            k_sgn1 = (-1)**bool(is1(site1, k))
            # state k_f1 component
            s_k_f1 = state[k_f1].item()

            # single site expectation values (x,y,z)
            evs[site1, 0] += np.conj(s_k_f1) * s_k
            evs[site1, 1] += 1j * k_sgn1 * np.conj(s_k_f1) * s_k
            evs[site1, 2] += k_sgn1 * abs(s_k)**2
            # two-site evs at different indices
            # (same indices can be calculated from sigle site evs only)
            for site2 in range(size):
                if site2 != site1:
                    # k with site1 and site2 flipped
                    k_f12 = k_f1 ^ (1 << site2)
                    # state k_f12 component
                    s_k_f12 = state[k_f12].item()
                    # 1 or -1 depending on whether k_f1 has a 0 or 1 on site2
                    k_f1_sgn2 = (-1)**bool(is1(site2, k_f1))
                    # k with site2 flipped
                    k_f2 = k ^ (1 << site2)
                    # state k_f2 component
                    s_k_f2 = state[k_f2].item()
                    # 1 or -1 depending on whether k has a 0 or 1 on site2
                    k_sgn2 = (-1)**bool(is1(site2, k))

                    # xx
                    evs2[site2, site1, 0, 0] += np.conj(s_k_f12) * s_k
                    # xy
                    evs2[site2, site1, 0, 1] += 1j * k_sgn1 * \
                        np.conj(s_k_f12) * s_k
                    # xz
                    evs2[site2, site1, 0, 2] += k_sgn1 * np.conj(s_k_f2) * s_k
                    # yx
                    evs2[site2, site1, 1, 0] += 1j * k_f1_sgn2 * \
                        np.conj(s_k_f12) * s_k
                    # yy
                    evs2[site2, site1, 1, 1] -= k_sgn1 * k_f1_sgn2 * \
                        np.conj(s_k_f12) * s_k
                    # yz
                    evs2[site2, site1, 1, 2] += 1j * k_sgn1 * k_sgn2 * \
                        np.conj(s_k_f2) * s_k
                    # zx
                    evs2[site2, site1, 2, 0] += k_f1_sgn2 * \
                        np.conj(s_k_f1) * s_k
                    # zy
                    evs2[site2, site1, 2, 1] += 1j * k_sgn1 * k_f1_sgn2 * \
                        np.conj(s_k_f2) * s_k
                    # zz
                    evs2[site2, site1, 2, 2] += k_sgn1 * k_sgn2 * abs(s_k)**2

        # 2-point corr on the same site can be calculated from sigle-site evs
        evs2[site1, site1, 0, 0] = 1
        evs2[site1, site1, 0, 1] = 1j * evs[site1, 2]
        evs2[site1, site1, 0, 2] = -1j * evs[site1, 1]
        evs2[site1, site1, 1, 0] = -1j * evs[site1, 2]
        evs2[site1, site1, 1, 1] = 1
        evs2[site1, site1, 1, 2] = 1j * evs[site1, 0]
        evs2[site1, site1, 2, 0] = 1j * evs[site1, 1]
        evs2[site1, site1, 2, 1] = -1j * evs[site1, 0]
        evs2[site1, site1, 2, 2] = 1
    # generate the L**2 matrices of 3x3 dimension that contain all directions
    # of correlators given site1&site2

    corr = np.zeros((size, size, 3, 3), dtype=np.complex128)
    for site2 in range(size):
        for site1 in range(size):
            for ax2 in range(3):
                for ax1 in range(3):
                    corr[site2, site1, ax2, ax1] = \
                        evs2[site2, site1, ax2, ax1] - (evs[site1, ax1]
                                                        * evs[site2, ax2])
    return corr
