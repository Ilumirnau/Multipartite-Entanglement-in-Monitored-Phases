import numpy as np
from numba import njit, prange
from scipy import stats


# STATEVECTOR SIMULATION AND CORRELATIONS


def is1(index: int, d: int) -> int:
    """
    Check if the bit at a specific index is set to 1 in the binary form of d.
    Args:
        index (int): The position of the bit to check (0-based index, start
                     right-most bit).
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
            The index represents the bit position (right-most bit has idx 0),
            and the value represents the bit value (0 or 1).
    N (int): The integer to which the bits will be assigned.
    Returns:
    int: The new integer value after the bits have been assigned.
    Example:
    >>> bit_assign([(2, 1)], 5)
    40
    >>> bit_assign([(1, 0), (3, 1)], 8)
    18
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


def measure_x(index: int, state: np.ndarray, size: int) -> np.ndarray:
    """
    Measure the X operator on a specific qubit in the given state.
    Parameters:
    index (int): The index of the qubit to measure.
    state (np.ndarray): The state vector to measure.
    size (int): The size of the system.
    Returns:
    np.ndarray: The state vector after measuring the X operator.
    """

    p_cross = 0
    for i, c_i in enumerate(state):
        j = i ^ (1 << index)
        p_cross += c_i.conjugate() * state[j]
    p_plus = 1/2 * (1 + p_cross)
    p_plus = np.real(p_plus).item()
    p_minus = 1 - p_plus

    meas_outcome = np.random.rand()
    if meas_outcome < p_plus:
        proj = 1/2 * np.ones((2, 2))
        norm = p_plus
    else:
        proj = np.array([[1, -1], [-1, 1]]) / 2
        norm = p_minus

    new_state = np.zeros_like(state)
    for z in range(2**(size-1)):
        row0 = bit_assign([(index, 0)], z)
        row1 = bit_assign([(index, 1)], z)

        new_state[[row0, row1]] = proj @ state[[row0, row1]]
    new_state /= np.sqrt(norm)

    assert np.isclose(np.vdot(new_state, new_state), 1)
    return new_state


def measure_z(index: int, state: np.ndarray, size: int) -> np.ndarray:
    """
    Measure the Z operator on a specific qubit in the given state.
    Parameters:
    index (int): The index of the qubit to measure.
    state (np.ndarray): The state vector to measure.
    size (int): The size of the system.
    Returns:
    np.ndarray: The state vector after measuring the Z operator.
    """

    p0 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                if not is1(index, i)])
    p1 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                if is1(index, i)])

    meas_outcome = np.random.rand()
    if meas_outcome < p0:
        proj = np.array([[1, 0], [0, 0]])
        norm = p0
    else:
        proj = np.array([[0, 0], [0, 1]])
        norm = p1

    new_state = np.zeros_like(state)
    for z in range(2**(size-1)):
        row0 = bit_assign([(index, 0)], z)
        row1 = bit_assign([(index, 1)], z)

        new_state[[row0, row1]] = proj @ state[[row0, row1]]
    new_state /= np.sqrt(norm)

    assert np.isclose(np.vdot(new_state, new_state), 1)
    return new_state


def measure_u1(index1: int, index2: int,
               state: np.ndarray, size: int) -> np.ndarray:
    "Update the state according to the measurement on sites index1 & index2"

    # probabilities of each outcome given a state
    p00 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                  if not is1(index1, i) and not is1(index2, i)])
    p11 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                  if is1(index1, i) and is1(index2, i)])
    p10 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                  if is1(index1, i) and not is1(index2, i)])
    p01 = np.sum([abs(c_i)**2 for i, c_i in enumerate(state)
                  if not is1(index1, i) and is1(index2, i)])

    p_cross = 0

    for z in range(2**(size-2)):
        # i 10, j 01
        i = bit_assign(((index1, 1), (index2, 0)), z)
        i_flip = bit_assign(((index1, 0), (index2, 1)), z)
        p_cross += 2 * np.real(state[i_flip].conjugate() * state[i])

    p_0 = p00 + p11
    p_plus = 1/2 * (p01 + p10 + p_cross)
    p_plus = np.real(p_plus).item()
    p_minus = 1/2 * (p01 + p10 - p_cross)
    # p_minus = 1 - p_plus - p00 - p11
    # print(p_plus+p_minus+p_0) -> should always be 1

    # update the state
    meas_outcome = np.random.rand()
    if meas_outcome < p_plus:
        # project qubits to psi+
        proj = np.zeros((4, 4))
        proj[1, 1] = 1
        proj[1, 2] = 1
        proj[2, 1] = 1
        proj[2, 2] = 1
        proj = 1/2 * proj
        norm = p_plus
        # print('+')

    elif meas_outcome < p_plus + p_minus:
        # project qubits to psi-
        proj = np.zeros((4, 4))
        proj[1, 1] = 1
        proj[1, 2] = -1
        proj[2, 1] = -1
        proj[2, 2] = 1
        proj = 1/2 * proj
        norm = p_minus
        # print('-')

    else:
        # project qubits to {|00>,|11>} subspace
        proj = np.zeros((4, 4))
        proj[0, 0] = 1
        proj[3, 3] = 1
        norm = p_0
        # print('0')

    new_state = np.zeros_like(state)
    for z in range(2**(size-2)):
        row00 = bit_assign(((index1, 0), (index2, 0)), z)
        row01 = bit_assign(((index1, 0), (index2, 1)), z)
        row10 = bit_assign(((index1, 1), (index2, 0)), z)
        row11 = bit_assign(((index1, 1), (index2, 1)), z)

        new_state[[row00, row01, row10, row11]] = proj @ state[[row00, row01,
                                                                row10, row11]]
    new_state /= np.sqrt(norm)

    assert np.isclose(np.vdot(new_state, new_state), 1)
    return new_state


def random_u1() -> np.ndarray:
    """
    Generate a random U(1) operation
    Returns:
    np.ndarray: The U(1) operation.
    """
    # Generate a random angle for the U(1) operation
    alpha = np.random.uniform(0, 2 * np.pi)
    beta = np.random.uniform(0, 2 * np.pi)
    gamma = np.random.uniform(0, 2 * np.pi)
    eta = np.random.uniform(0, 2 * np.pi)
    xi = np.random.uniform(0, 2 * np.pi)

    t = np.random.uniform(0, 1)  # |sin(psi)|^2
    sin_psi = np.sqrt(t)
    cos_psi = np.sqrt(1 - t)

    # Create the U(1) operator
    u1_operator = np.zeros((4, 4), dtype=complex)
    u1_operator[0, 0] = np.exp(1j * alpha)
    u1_operator[1, 1] = np.exp(1j * (beta+eta)) * cos_psi
    u1_operator[1, 2] = np.exp(1j * (beta+xi)) * sin_psi
    u1_operator[2, 1] = np.exp(1j * (beta-xi)) * -sin_psi
    u1_operator[2, 2] = np.exp(1j * (beta-eta)) * cos_psi
    u1_operator[3, 3] = np.exp(1j * gamma)

    return u1_operator


def random_u1_h() -> np.ndarray:
    """
    Generate a random U(1) operation from Haar random
    Returns:
    np.ndarray: The U(1) 4x4 matrix.
    """
    alpha = np.random.uniform(0, 2 * np.pi)
    beta = np.random.uniform(0, 2 * np.pi)

    u1_operator = np.zeros((4, 4), dtype=complex)
    u1_operator[0, 0] = np.exp(1j * alpha)
    u1_operator[1:3, 1:3] = stats.unitary_group.rvs(2)
    u1_operator[3, 3] = np.exp(1j * beta)

    return u1_operator


def U1_ij(index1: int, index2: int, state: np.ndarray,
          size: int,) -> np.ndarray:
    """
    Applies a random 4x4 unitary transformation to the specified qubits in the
    given quantum state, which is U(1) symmetric.
    Parameters:
    index1 (int): The index of the first qubit.
    index2 (int): The index of the second qubit.
    state (numpy.ndarray): The current state vector of the quantum system.
    size (int): The number of qubits in the system.
    Returns:
    numpy.ndarray: The new state vector after applying the unitary gate.
    """

    new_state = np.zeros_like(state)
    hg = random_u1_h()

    for z in range(2**(size-2)):
        row00 = bit_assign(((index1, 0), (index2, 0)), z)
        row01 = bit_assign(((index1, 0), (index2, 1)), z)
        row10 = bit_assign(((index1, 1), (index2, 0)), z)
        row11 = bit_assign(((index1, 1), (index2, 1)), z)

        new_state[[row00, row01, row10, row11]] = hg @ state[[row00, row01,
                                                              row10, row11]]
    return new_state


def U_ij(index1: int, index2: int, state: np.ndarray,
         size: int,) -> np.ndarray:
    """
    Applies a random 4x4 unitary transformation to the specified qubits in the
    given quantum state, which is U(1) symmetric.
    Parameters:
    index1 (int): The index of the first qubit.
    index2 (int): The index of the second qubit.
    state (numpy.ndarray): The current state vector of the quantum system.
    size (int): The number of qubits in the system.
    Returns:
    numpy.ndarray: The new state vector after applying the unitary gate.
    """

    new_state = np.zeros_like(state)
    hg = stats.unitary_group.rvs(2**2)

    for z in range(2**(size-2)):
        row00 = bit_assign(((index1, 0), (index2, 0)), z)
        row01 = bit_assign(((index1, 0), (index2, 1)), z)
        row10 = bit_assign(((index1, 1), (index2, 0)), z)
        row11 = bit_assign(((index1, 1), (index2, 1)), z)

        new_state[[row00, row01, row10, row11]] = hg @ state[[row00, row01,
                                                              row10, row11]]
    return new_state


def apply_row_S(state: np.ndarray, px: float, pu: float,
                size: int, inverted: bool) -> np.ndarray:
    """
    Applies U(1) measurements, local-x measurements and random u(1) unitaries.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    probs = np.random.random(size)

    # sites to apply u(1) measurement
    sites_u1m = [i for i in range(size) if probs[i] > (px+pu)]
    # sites to apply projective x-measurements
    sites_zm = [i for i in range(size)
                if (probs[i] > pu and probs[i] < (pu+px))]
    # sites to apply random unitaries
    sites_u = [i for i in range(size) if probs[i] < pu]

    if inverted:
        for site in sites_u:
            state = U1_ij(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_x(site, state, size)
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
    else:
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_x(site, state, size)
        for site in sites_u:
            state = U1_ij(site, (site+1) % size, state, size)

    return state


def apply_row_NS(state: np.ndarray, px: float, pu: float,
                 size: int, inverted: bool) -> np.ndarray:
    """
    Applies U(1) measurements, local-x measurements and random unitaries.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    probs = np.random.random(size)

    # sites to apply u(1) measurement
    sites_u1m = [i for i in range(size) if probs[i] > (px+pu)]
    # sites to apply projective z-measurements
    sites_zm = [i for i in range(size)
                if (probs[i] > pu and probs[i] < (pu+px))]
    # sites to apply random unitaries
    sites_u = [i for i in range(size) if probs[i] < pu]

    if inverted:
        for site in sites_u:
            state = U_ij(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_x(site, state, size)
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
    else:
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_x(site, state, size)
        for site in sites_u:
            state = U_ij(site, (site+1) % size, state, size)

    return state


def brickwork_row(state: np.ndarray, px: float,
                  time: int, size: int) -> np.ndarray:
    """
    Applies U(1) measurements, and local-X measurements following
    a brickwork pattern.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    # Apply u(1) measurements with parity according to time step
    for i in range(time % 2, size, 2):
        state = measure_u1(i, (i+1) % size, state, size)

    # Apply local x-measurements
    probs = np.random.rand(size)
    for i in range(size):
        if probs[i] < px:
            # measure the qubit
            state = measure_x(i, state, size)
    return state


def apply_row_S_z(state: np.ndarray, px: float, pu: float,
                  size: int, inverted: bool) -> np.ndarray:
    """
    Applies U(1) measurements, local-z measurements and random u(1) unitaries.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    probs = np.random.random(size)

    # sites to apply u(1) measurement
    sites_u1m = [i for i in range(size) if probs[i] > (px+pu)]
    # sites to apply projective x-measurements
    sites_zm = [i for i in range(size)
                if (probs[i] > pu and probs[i] < (pu+px))]
    # sites to apply random unitaries
    sites_u = [i for i in range(size) if probs[i] < pu]

    if inverted:
        for site in sites_u:
            state = U1_ij(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_z(site, state, size)
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
    else:
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_z(site, state, size)
        for site in sites_u:
            state = U1_ij(site, (site+1) % size, state, size)

    return state


def apply_row_NS_z(state: np.ndarray, px: float, pu: float,
                   size: int, inverted: bool) -> np.ndarray:
    """
    Applies U(1) measurements, local-z measurements and random unitaries.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    probs = np.random.random(size)

    # sites to apply u(1) measurement
    sites_u1m = [i for i in range(size) if probs[i] > (px+pu)]
    # sites to apply projective z-measurements
    sites_zm = [i for i in range(size)
                if (probs[i] > pu and probs[i] < (pu+px))]
    # sites to apply random unitaries
    sites_u = [i for i in range(size) if probs[i] < pu]

    if inverted:
        for site in sites_u:
            state = U_ij(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_z(site, state, size)
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
    else:
        for site in sites_u1m:
            state = measure_u1(site, (site+1) % size, state, size)
        for site in sites_zm:
            state = measure_z(site, state, size)
        for site in sites_u:
            state = U_ij(site, (site+1) % size, state, size)

    return state


def brickwork_row_z(state: np.ndarray, px: float,
                    time: int, size: int) -> np.ndarray:
    """
    Applies U(1) measurements, and local-z measurements following
    a brickwork pattern.
    Parameters:
    state (np.ndarray): Initial state to which the unitary operation is applied
    px (float): Probability of applying a local measurement.
    pu (float): Probability of applying a random unitary.
    size (int): The system size.
    inverted (bool): If True, the gates are applied in the opposite direction.
    Returns:
    np.ndarray: The state after applying the unitary operations.
    """
    # Apply u(1) measurements with parity according to time step
    for i in range(time % 2, size, 2):
        state = measure_u1(i, (i+1) % size, state, size)

    # Apply local x-measurements
    probs = np.random.rand(size)
    for i in range(size):
        if probs[i] < px:
            # measure the qubit
            state = measure_z(i, state, size)
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

# SIMULATED ANNEALING, QUANTUM FISHER INFORMATION


@njit(fastmath=True, parallel=True)
def qfi(ns, C, size):
    fq = 0
    for i in prange(size):
        for j in prange(size):
            fq += ns[i] @ C[i, j] @ ns[j]
    return fq/size


@njit(fastmath=True)
def energy(phis, thetas, C, size):
    '''
    This function calculates the cost function of a particular configuration.
    In this case the cost function is minus the QFI density
    phis: List of angles
    thetas: List of angles
    C: np.array with the correlations in all directions and sites
    l = int size of the system
    '''
    n_x = np.cos(phis) * np.sin(thetas)
    n_y = np.sin(phis) * np.sin(thetas)
    n_z = np.cos(thetas)

    ns = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(size)]
    return -qfi(ns, C, size)


@njit(fastmath=True)
def ratio(E, E_pert, T):
    '''
    This function returns the decision ratio between the old and the new costs.
    E<E_pert will give r<1 (which is againist minimization)
    Te
    E: float old cost
    E_pert: float new cost
    T: temperature of the simulated annealing algorithm
    '''
    r = np.exp((E-E_pert)/T)
    return r


@njit(fastmath=True)
def met_phi(E, phis, thetas, idx, T, delta, C, size):
    '''
    Apply a  metropolis step on phi: randomly change a value and save the
    change with probability given by `ratio`.
    E: float current cost
    phis: np.array current phi angles
    thetas: np.array current theta angles
    idx: int index of phi to change
    delta: float step size of change
    C: np.array all correlations
    l: int system size
    '''
    new_phi = phis[idx]
    new_phi += (2 * np.random.random() - 1) * delta  # mean 0 variation
    new_phi -= 2 * np.pi * np.rint(new_phi/(2*np.pi))  # keeping phi in +-pi
    dphi = new_phi - phis[idx]
    phis[idx] += dphi

    E_pert = energy(phis, thetas, C, size)
    # change is unfavorable
    if E_pert > E:
        r = ratio(E, E_pert, T)

        z = np.random.random()
        if z < r:
            # accept
            E_new = E_pert
        else:
            # refuse and undo change
            E_new = E
            phis[idx] -= dphi
    # change is favorable
    else:
        # accept
        E_new = E_pert
    return E_new, phis


@njit(fastmath=True)
def met_theta(E, phis, thetas, idx, T, delta, C, size):
    '''
    Apply a  metropolis step on theta: randomly change a value and save the
    change with probability given by `ratio`.
    E: float current cost
    phis: np.array current phi angles
    thetas: np.array current theta angles
    idx: int index of theta to change
    delta: float step size of change
    C: np.array all correlations
    l: int system size
    '''
    dtheta = (2 * np.random.random() - 1) * delta  # mean 0 variation
    if abs(thetas[idx] + dtheta - np.pi/2) > np.pi/2:
        # if dtheta takes us away from the range [0, pi), flip sign to reverse
        dtheta = -1 * dtheta
    thetas[idx] += dtheta

    E_pert = energy(phis, thetas, C, size)
    # change is unfavorable
    if E_pert > E:
        r = ratio(E, E_pert, T)

        z = np.random.random()
        if z < r:
            # accept
            E_new = E_pert
        else:
            # refuse and undo change
            E_new = E
            thetas[idx] -= dtheta
    # change is favorable
    else:
        # accept
        E_new = E_pert
    return E_new, thetas


@njit(fastmath=True)
def monte_carlo(phis, thetas, T, n_iter, delta, C, size):
    '''
    Apply the monte carlo algorithm `n_iter` times at temperature `T` to find
    minimum energy and corresponding spin configuration.
    phis: np.array list of phi angles
    thetas: np.array list of theta angles
    T: temperature at which to perform the equilibration algorithm
    n_iter: int times to repeat the algorithm. If 0 stops at relative error T
    delta: float step size of the monte carlo algorithm
    C: np.array correlations of all sites in all directions
    l: int system size
    '''
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


@njit(fastmath=True)
def sim_annealing(schedule, repeats, C, size):
    '''
    Appply simulated annealing to find vectors that maximize qfi (mnimize -qfi)
    Schedule: np.array specifies #steps at each temperature, temperature
            considered, step delta
    repeats: int amount of times to repeat the optimization algorithm
    C: np.array corelations of all sites in all directions
    l: int system size
    '''
    num_Ts = np.shape(schedule)[0]
    qfi = 0
    error = 0
    for rep in range(repeats):
        # print(f'Optimizing repetition {rep}')
        phis = np.pi * (2 * np.random.random_sample(size) - np.ones(size))
        thetas = np.pi * np.random.random_sample(size)
        for i in range(num_Ts):
            n_iter, T, delta = schedule[i]
            E_GS, phis, thetas, err = monte_carlo(phis, thetas, T,
                                                  n_iter, delta, C, size)
        if -E_GS > qfi:
            error = err
            qfi = -E_GS

    return qfi, error
