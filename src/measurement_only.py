# ------------------------------------------------------------------------------
"""This script simulates a measurement-only quantum circuit and calculates the
quantum Fisher information (QFI) of the resulting state in the context of a
color code.
Functions:
----------
nextS(S):
    Finds the least nonzero integer in the array S.
mz(S, i):
    Measures the z operator on site i, setting S[i] to 0.
mxx(S, i, j):
    Measures the xx operator on sites i and j, updating S accordingly.
layer(S, pz):
    Applies a projective measurement layer to the state S, where pz is the
    probability of measuring z.
cc_qfi(S):
    Calculates the quantum Fisher information (QFI) of the state S in the
    color code context.
Main Execution:
---------------
The script takes two command-line arguments: L (sys. size) and seed (random).
It initializes the state S with zeros, applies the measurement layers, and
calculates the QFI for different probabilities pz.
The results are saved to a file in the 'data/cc_qfi/' directory.
"""
# ------------------------------------------------------------------------------
import numpy as np
from numba import njit

# ------------------------------------------------------------------------------


@njit(fastmath=True)
def nextS(S: list) -> int:
    """Find the least nonzero integer not present in the input list S.

    Parameters:
    S (list of int): A list of integers.

    Returns:
    int: The smallest nonzero integer that is not in the list S.
    """

    N = len(S)
    inS = np.unique(S)
    for s in range(1, N):
        if s not in inS:
            return s


@njit(fastmath=True)
def mz(S: list, i: int) -> list:
    """Measure the z-component on site i and set it to 0.

    Parameters:
    S (list): A list representing the state of the system.
    i (int): The index of the site to measure.

    Returns:
    list: The updated state of the system with the z-component at site i
    set to 0.
    """

    S[i] = 0
    return S


@njit(fastmath=True)
def mxx(S: list, i: int, j: int) -> list:
    """Perform a measurement operation on the list S at indices i and j.
    This function updates the list S based on the values at indices i and j.
    The rules for updating are as follows:
    - If both S[i] and S[j] are 0, set both S[i] and S[j] to the next state.
    - If S[i] is non-zero and S[j] is zero, set S[j] to S[i].
    - If S[i] is zero and S[j] is non-zero, set S[i] to S[j].
    - If both S[i] and S[j] are non-zero, set all elements in S that are equal
      to either S[i] or S[j] to the minimum of S[i] and S[j].
    Args:
        S (list): The list of states.
        i (int): The index of the first element to measure.
        j (int): The index of the second element to measure.
    Returns:
        list: The updated list of states.
    """

    si = S[i]
    sj = S[j]
    if not si+sj:
        next_s = nextS(S)
        S[i] = S[j] = next_s
    elif (si and (not sj)):
        S[j] = S[i]
    elif ((not si) and sj):
        S[i] = S[j]
    elif si and sj:
        s = min(si, sj)
        for idx, val in enumerate(S):
            if val == si or val == sj:
                S[idx] = s
    return S


@njit(fastmath=True)
def layer(S: list, pz: float) -> list:
    """
    apply projective measurement layer, where pz is the probability of
    measuring z
    """
    N = len(S)
    original_sites = np.arange(N)
    sites = np.random.permutation(original_sites)
    probs = np.random.random(N)

    sz = sites[probs < pz]
    sxx = sites[probs >= pz]

    for i in sxx:
        S = mxx(S, i, (i+1) % N)
    for j in sz:
        S = mz(S, j)
    return S


def cc_qfi(S: list) -> float:
    """
    calculate the qfi of the state in the color code context
    """
    N = len(S)
    vals, counts = np.unique(S, return_counts=True)
    qfi = 0
    for val, count in zip(vals, counts):
        if val:
            qfi += int(count)**2
        else:
            qfi += int(count)
    return qfi/N
