# ------------------------------------------------------------------------------
"""
This module implements a simulated annealing algorithm to minimize the energy
function (-1 * quantum Fisher information density) for a given configuration.
Functions:
    qfi(ns, C, l):
        Calculate the QFI density for a given configuration.
    energy(phis, thetas, C, l):
        Calculate the cost function (negative QFI density)
          for a given configuration.
    ratio(E, E_pert, T):
        Calculate the decision ratio between the old and new costs.
    met_phi(E, phis, thetas, idx, T, delta, C, l):
        Apply a Metropolis step on phi angles and update the configuration
          based on the decision ratio.
    met_theta(E, phis, thetas, idx, T, delta, C, l):
        Apply a Metropolis step on theta angles and update the configuration
          based on the decision ratio.
    monte_carlo(phis, thetas, T, n_iter, delta, C, l):
        Apply the Metropolis algorithm to find the equilibrium configuration
          at a given temperature.
    sim_annealing(schedule, repeats, C, l):
        Apply simulated annealing to find vectors that maximize the QFI dens.
"""

# ---------------------------------IMPORT LIBRARIES----------------------------
from numba import njit, prange
import numpy as np

# ------------------------------------------------------------------------------


@njit(fastmath=True, parallel=True)
def qfi(ns: list, C: np.ndarray, size: int) -> float:
    """
    Calculate the quantum Fisher information (QFI) for a given
      set of states and a covariance matrix.
    Args:
        ns (list): A list of state vectors.
        C (np.ndarray): A covariance matrix.
        size (int): The size of the system (number of qubits).
    Returns:
        float: The calculated quantum Fisher information (QFI).
    """

    fq = 0
    for i in prange(size):
        for j in prange(size):
            fq += ns[i] @ C[i, j] @ ns[j]
    return fq/size


@njit(fastmath=True)
def energy(phis: np.ndarray, thetas: np.ndarray,
           C: np.ndarray, size: int) -> float:
    """
    Calculate the energy of a system based on given angles and coupling matrix.
    Parameters:
    phis (array-like): Array of azimuthal angles (in radians).
    thetas (array-like): Array of polar angles (in radians).
    C (array-like): Coupling matrix.
    size (int): Number of elements in the system.
    Returns:
    float: The negative QFI of the system.
    """

    n_x = np.cos(phis) * np.sin(thetas)
    n_y = np.sin(phis) * np.sin(thetas)
    n_z = np.cos(thetas)

    ns = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(size)]
    return -qfi(ns, C, size)


@njit(fastmath=True)
def ratio(E: float, E_pert: float, T: float) -> float:
    """
    Calculate the ratio of the exponent of
     the energy difference over temperature.
    Parameters:
    E (float): The current energy.
    E_pert (float): The perturbed energy.
    T (float): The temperature.
    Returns:
    float: The ratio calculated as exp((E - E_pert) / T).
    """

    r = np.exp((E-E_pert)/T)
    return r


@njit(fastmath=True)
def met_phi(E: float, phis: np.ndarray, thetas: np.ndarray, idx: int,
            T: float, delta: float, C: np.ndarray, size: int) -> tuple:
    """
    Perform a Metropolis update on the phi angles in simulated annealing.
    Parameters:
    E (float): Current energy of the system.
    phis (numpy.ndarray): Array of phi angles.
    thetas (numpy.ndarray): Array of theta angles.
    idx (int): Index of the phi angle to be updated.
    T (float): Current temperature of the system.
    delta (float): Maximum change allowed for the phi angle.
    C (numpy.ndarray): Coupling matrix.
    size (int): Size of the system.
    Returns:
    tuple: Updated energy and phi angles.
    """

    new_phi = phis[idx]
    new_phi += (2 * np.random.random() - 1) * delta  # mean 0 variation
    new_phi -= 2 * np.pi * np.rint(new_phi/(2*np.pi))  # keeping phi<=|pi|
    dphi = new_phi - phis[idx]
    phis[idx] += dphi

    E_pert = energy(phis, thetas, C, size)

    if E_pert > E:  # change is unfavorable
        r = ratio(E, E_pert, T)

        z = np.random.random()
        if z < r:  # accept
            E_new = E_pert
        else:  # refuse and undo change
            E_new = E
            phis[idx] -= dphi
    else:  # change is favorable, accept
        E_new = E_pert
    return E_new, phis


@njit(fastmath=True)
def met_theta(E: float, phis: np.ndarray, thetas: np.ndarray, idx: int,
              T: float, delta: float, C: np.ndarray, size: int) -> tuple:
    """
    Perform a Metropolis update on the theta angles in simulated annealing.
    Parameters:
    E (float): Current energy of the system.
    phis (numpy.ndarray): Array of phi angles.
    thetas (numpy.ndarray): Array of theta angles.
    idx (int): Index of the theta angle to be updated.
    T (float): Current temperature of the system.
    delta (float): Maximum change allowed for the theta angle.
    C (numpy.ndarray): Coupling matrix.
    size (int): Size of the system.
    Returns:
    tuple: Updated energy and theta angles.
    """

    dtheta = (2 * np.random.random() - 1) * delta  # mean 0 variation
    if abs(thetas[idx] + dtheta - np.pi/2) > np.pi/2:
        # if dtheta takes us away from the range [0, pi), reverse it
        dtheta = -1 * dtheta
    thetas[idx] += dtheta

    E_pert = energy(phis, thetas, C, size)
    if E_pert > E:  # change is unfavorable
        r = ratio(E, E_pert, T)

        z = np.random.random()
        if z < r:  # accept
            E_new = E_pert
        else:  # refuse and undo change
            E_new = E
            thetas[idx] -= dtheta
    else:  # change is favorable, accept
        E_new = E_pert
    return E_new, thetas


@njit(fastmath=True)
def monte_carlo(phis: np.ndarray, thetas: np.ndarray, T: float,
                n_iter: int, delta: float, C: np.ndarray, size: int) -> tuple:
    """
    Perform a Monte Carlo simulation to find the ground state energy and
    corresponding angles.
    Parameters:
    phis (array-like): Initial values of the phi angles.
    thetas (array-like): Initial values of the theta angles.
    T (float): Temperature or threshold for the error.
    n_iter (int): Number of iterations to perform.
    delta (float): Step size for the Metropolis algorithm.
    C (array-like): Coupling matrix.
    size (int): Size of the system.
    Returns:
    tuple: A tuple containing:
        - E_GS (float): Ground state energy.
        - phis (array-like): Optimized phi angles.
        - thetas (array-like): Optimized theta angles.
        - error (float): Final error value.
    """

    E_GS = energy(phis, thetas, C, size)
    E = E_GS
    error = 1
    for _ in range(n_iter):
        idx = np.random.randint(size)
        E, phis = met_phi(E, phis, thetas, idx, T, delta, C, size)
        E, thetas = met_theta(E, phis, thetas, idx, T, delta, C, size)
        if E < E_GS:
            error = abs(E_GS - E)
            E_GS = E
    return E_GS, phis, thetas, error


@njit(fastmath=True, parallel=True)
def sim_annealing(schedule: np.ndarray, repeats: int,
                  C: np.ndarray, size: int) -> tuple:
    """
    Perform simulated annealing, find the optimal parameters of a given system.
    Parameters:
    schedule (ndarray): A 2D array where each row represents a step
                        in the annealing schedule.
                        Each row should contain three values:
                         number of iterations (n_iter),
                         temperature (T), and update size (delta).
    repeats (int): The number of times the annealing process is repeated.
    C (float): A constant parameter used in the Monte Carlo simulation.
    size (int): The size vectors to optimize (qubits in the system).
    Returns:
    tuple: A tuple with the optimal QFI density and last step improvement
              obtained during the annealing process.
    """

    num_Ts = np.shape(schedule)[0]
    qfi = 0
    error = 0
    e_GS = 0
    for _ in range(repeats):
        phis = np.pi * (2 * np.random.random_sample(size) - np.ones(size))
        thetas = np.pi * np.random.random_sample(size)
        for i in range(num_Ts):
            n_iter, T, delta = schedule[i]
            e_GS, phis, thetas, err = monte_carlo(phis, thetas,
                                                  T, n_iter, delta, C, size)
        if (-e_GS) > qfi:
            error = err
            qfi = -e_GS

    return qfi, error
