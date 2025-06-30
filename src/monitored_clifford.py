# ------------------------------------------------------------------------------
""" This module provides functions for simulating quantum states using
Clifford gates and measuring their properties using the Stim library.
Functions:
    layer_clifford_brickwork(psi: stim.TableauSimulator, depth: int) -> None:
    layer_measurement(psi: stim.TableauSimulator, pm: float) -> None:
    corr_matrix(psi: stim.TableauSimulator) -> np.ndarray:
"""


# ------------------------------------------------------------------------------

import numpy as np
import stim

# ------------------------------------ FUNCTIONS FOR SIMULATION ---------------


def layer_clifford_brickwork(psi: stim.TableauSimulator, depth: int) -> None:
    """
    Applies a layer of two-body nearest-neighboring Clifford gates onto the
    given tableau state.
    Parameters:
    psi (stim.TableauSimulator): The tableau state to which the Clifford gates
    will be applied.
    depth (int): Determines the tilting of the layer. If even, the layer
    starts from the first qubit; if odd, it starts from the second qubit.
    Returns:
    None
    """
    N = psi.num_qubits
    v = depth % 2
    for i in range(v, N+v, 2):
        U = stim.Tableau.random(2)
        psi.do_tableau(U, [i, (i+1) % N])


def layer_measurement(psi: stim.TableauSimulator, pm: float) -> None:
    """
    Perform a layer of measurements on the given quantum state.
    Parameters:
    psi (stim.TableauSimulator): The quantum state simulator on which
    measurements are performed.
    pm (float): The probability of measuring each qubit.
    Returns:
    None
    """

    N = psi.num_qubits
    for i in range(N):
        if (np.random.rand() < pm):
            psi.measure(i)


def corr_matrix(psi: stim.TableauSimulator) -> np.ndarray:
    """
    Calculate the correlator matrix C_full for a given tableau state.
    The correlator c_ab is defined as:
        c_ab = 1/2 <{s_a, s_b}> - <s_a><s_b>
    where s_i are the total spin operators sum_j sigma_j^i.
    This results in a real symmetric matrix.
    Parameters:
    psi (stim.TableauSimulator): The tableau state simulator.
    Returns:
    np.ndarray: A 4-dimensional array of shape (L, L, 3, 3) storing the
                correlator matrix, where L is the number of qubits, and
                the last two dimensions correspond to the Pauli axes (X, Y, Z).
    """

    L = psi.num_qubits

    pauli_axes = ['X', 'Y', 'Z']

    C_full = np.zeros((L, L, 3, 3), dtype=float)

    peeks = [lambda psi, i: psi.peek_x(i),
             lambda psi, i: psi.peek_y(i),
             lambda psi, i: psi.peek_z(i)]

    for a, alpha in enumerate(pauli_axes):
        for b, beta in enumerate(pauli_axes):
            for site0 in range(L):
                e0 = peeks[a](psi, site0)
                for site1 in range(L):
                    e1 = peeks[b](psi, site1)
                    if site1 == site0:
                        C_full[site0, site1, a, b] = int(a == b) - e0 * e1
                    else:
                        P0 = stim.PauliString('I' * L)
                        P0[site0] = alpha
                        P0[site1] = beta
                        C_full[site0, site1, a, b] = \
                            psi.peek_observable_expectation(P0) - e0 * e1

    assert np.isclose(0, np.linalg.norm(np.sum(C_full, axis=(0, 1))
                                        - np.sum(C_full, axis=(0, 1)).T))
    return C_full
