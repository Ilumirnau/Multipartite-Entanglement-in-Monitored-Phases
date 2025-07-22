import stim
import numpy as np
import sys

def symmetry_preserving_us(size: int):
    '''
    This function generates a list of symmetry preserving clifford operations of a given size.
    
    Parameters
    ----------
    size : int
        number of qubits.

    Returns
    -------
    List of stim.Tableau objects

    '''
    G = stim.PauliString("ZZ").to_tableau()
    Us = []
    for u in stim.Tableau.iter_all(size, unsigned=True):
        if u*G*u.inverse() == G:
            Us.append(u)
    return Us

def triphase_layer(psi: stim.TableauSimulator, Us: list, pz, pu):
    '''
    This function generates a sigle time step of the three-phase diagram

    Parameters
    ----------
    psi : stim.TableauSimulator
        state of the system, it is evolved and changed by the function
    Us : list
        list of 2-qubit unitaries that respect the symmetry ZZ
    pz : TYPE
        Probability of measuring local Z.
    pu : TYPE
        Probability of applying a randomly sampled U from the list Us.

    Returns
    -------
    None.

    '''
    N = psi.num_qubits
    probs = np.random.random(N)
    sites = np.random.permutation(range(N))
    
    # first we apply the entangling measurements XX
    xx = list(sites[(probs>=(pz+pu))])
    nxs = [[], [], []]

    # add numba speed-up?
    for i in xx:
        if i not in nxs[0] and (i+1)%N not in nxs[0]:
            nxs[0].append(i)
            nxs[0].append((i+1)%N)
        elif i not in nxs[1] and (i+1)%N not in nxs[1]:
            nxs[1].append(i)
            nxs[1].append((i+1)%N)
        else:
            nxs[2].append(i)
            nxs[2].append((i+1)%N)
    
    ixs = [" ".join(map(str, nx)) for nx in nxs if nx]
    for ix in ixs:
        psi.do_circuit(stim.Circuit(f'MXX {ix}'))
    
    # then we apply the local measurements Z
    iz = ' '.join(map(str, list(sites[probs<pz])))
    psi.do_circuit(stim.Circuit(f'M {iz}'))
    
    # finally we apply the unitaries
    ius = sites[(probs>=pz) & (probs<(pz+pu))]
    us = np.random.choice(Us, len(ius))
    for site, u in zip(ius, us):
        psi.do_tableau(u, [site, (site+1)%N])

    
def corr_matrix(psi: stim.TableauSimulator):
    '''
    Calculates the correlator c_ab = 1/2 <{s_a, s_b}> - <s_a><s_b>
    This is a real symmetric matrix
    psi: tableau state
    '''
    N = psi.num_qubits
    pauli_axes = ['X', 'Y', 'Z']
    C_full = np.zeros((N, N, 3, 3), dtype = float)
    peeks = [lambda psi,i: psi.peek_x(i), lambda psi,i: psi.peek_y(i), lambda psi,i: psi.peek_z(i)]
    
    for a, alpha in enumerate(pauli_axes):
        for b, beta in enumerate(pauli_axes):
            for site0 in range(N):
                e0 = peeks[a](psi, site0)
                for site1 in range(N):
                    e1 = peeks[b](psi,site1)
                    if site1==site0:
                        C_full[site0, site1, a, b] = int(a==b) - e0 * e1
                    else:
                        P0 = stim.PauliString('I' * N)
                        P0[site0] = alpha
                        P0[site1] = beta
                        C_full[site0, site1, a, b] = psi.peek_observable_expectation(P0) - e0 * e1
    assert np.isclose(0, np.linalg.norm(np.sum(C_full, axis=(0,1)) - np.sum(C_full, axis=(0,1)).T))
    return C_full


from numba import njit, prange
import os

#goal of the algorithm: minimze energy function (in this case minus fisher info density)
@njit(fastmath=True, parallel=True)
def qfi(ns, C, l):
    fq = 0
    for i in prange(l):
        for j in prange(l):
            fq += ns[i] @ C[i, j] @ ns[j]
    return fq/l


@njit(fastmath=True)
def energy(phis, thetas, C, l):
    '''
    This function calculates the cost function of a particular configuration.
    In this case the cost function is minus the quantum fisher information density
    phis: List of angles
    thetas: List of angles
    C: np.array with the correlations in all directions and sites
    l = int size of the system
    '''
    n_x = np.cos(phis) * np.sin(thetas)
    n_y = np.sin(phis) * np.sin(thetas)
    n_z = np.cos(thetas)
    
    ns = [np.array([n_x[i], n_y[i], n_z[i]]) for i in range(l)]
    return -qfi(ns, C, l)

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
def met_phi(E, phis, thetas, idx, T, delta, C, l):
    '''
    Apply a  metropolis step on phi: randomly change a value and save the change with
    probability given by `ratio`.
    E: float current cost
    phis: np.array current phi angles
    thetas: np.array current theta angles
    idx: int index of phi to change
    delta: float step size of change
    C: np.array all correlations
    l: int system size
    '''
    new_phi = phis[idx]
    new_phi += (2 * np.random.random() -1) * delta #mean 0 variation
    new_phi -= 2 * np.pi * np.rint(new_phi/(2*np.pi)) #keeping phi between -pi and +pi
    dphi = new_phi - phis[idx]
    phis[idx] += dphi
    
    E_pert = energy(phis, thetas, C, l)
    #change is unfavorable
    if E_pert > E:
        r =ratio(E, E_pert, T)
        
        z = np.random.random()
        if z<r:
            #accept
            E_new = E_pert
        else:
            #refuse and undo change
            E_new = E
            phis[idx] -= dphi
    #change is favorable
    else:
        #accept
        E_new = E_pert
    return E_new, phis

@njit(fastmath=True)
def met_theta(E, phis, thetas, idx, T, delta, C, l):
    '''
    Apply a  metropolis step on theta: randomly change a value and save the change with
    probability given by `ratio`.
    E: float current cost
    phis: np.array current phi angles
    thetas: np.array current theta angles
    idx: int index of theta to change
    delta: float step size of change
    C: np.array all correlations
    l: int system size
    '''
    dtheta = (2 * np.random.random() -1) * delta #mean 0 variation
    if abs(thetas[idx] + dtheta - np.pi/2) > np.pi/2:
        #if dtheta takes us away from the range [0, pi), flip its sign to reverse it
        dtheta = -1*dtheta 
    thetas[idx] += dtheta
    
    E_pert = energy(phis, thetas, C, l)
    #change is unfavorable
    if E_pert > E:
        r =ratio(E, E_pert, T)
        
        z = np.random.random()
        if z<r:
            #accept
            E_new = E_pert
        else:
            #refuse and undo change
            E_new = E
            thetas[idx] -= dtheta
    #change is favorable
    else:
        #accept
        E_new = E_pert
    return E_new, thetas

@njit(fastmath=True)
def monte_carlo(phis, thetas, T, n_iter, delta, C, l):
    '''
    Apply the monte carlo algorithm `n_iter` times at temperature `T` to find minimum 
    energy and corresponding spin configuration.
    phis: np.array list of phi angles
    thetas: np.array list of theta angles
    T: temperature at which to perform the equilibration algorithm
    n_iter: int times to repeat the algorithm. If 0 it stops at relative error T
    delta: float step size of the monte carlo algorithm
    C: np.array correlations of all sites in all directions
    l: int system size
    '''
    E_GS = energy(phis, thetas, C, l)
    E = E_GS
    error = 1
    for _ in range(n_iter):
        idx = np.random.randint(l)
        E, phis = met_phi(E, phis, thetas, idx, T, delta, C, l)
        E, thetas = met_theta(E, phis, thetas, idx, T, delta, C, l)
        if E<E_GS:
            error = abs(E_GS - E)
            E_GS = E
    return E_GS, phis, thetas, error

@njit(fastmath=True)
def sim_annealing(schedule, repeats, C, l):
    '''
    Appply simulated annealing to find vectors that maximize qfi (mnimize -qfi).
    Schedule: np.array specifies #steps at each temperature, temperature considered, step delta
    repeats: int amount of times to repeat the optimization algorithm
    C: np.array corelations of all sites in all directions
    l: int system size
    '''
    num_Ts = np.shape(schedule)[0]
    qfi = 0
    error = 0
    for rep in range(repeats):
        phis = np.pi * (2 * np.random.random_sample(l) - np.ones(l))
        thetas = np.pi * np.random.random_sample(l)
        for i in range(num_Ts):
            n_iter, T, delta = schedule[i]
            E_GS, phis, thetas, err = monte_carlo(phis, thetas, T, n_iter, delta, C, l)
        if -E_GS > qfi:
            error = err
            qfi = -E_GS
            
    return qfi, error


#annealing schedule: number of iterations at T, the temperature, delta (angle variation parameter)
schedule = np.array([[4000,1.,np.pi/1.5],[3500,0.8,np.pi/1.5],[3500,0.6,np.pi/1.5],[3500,0.4,np.pi/1.5],[3500,0.2,np.pi/1.5],
                  [5000,0.1,np.pi/2],[5000,0.08,np.pi/2],[5000,0.06,np.pi/2],[5000,0.04,np.pi/2],[5000,0.02,np.pi/4],
                  ])
repeats = 5


L, seed = sys.argv[1:]
L, seed = int(L), int(seed)
np.random.seed(seed * 13)
t = 4 * L

Us = symmetry_preserving_us(2)

ps = []

if L<65:
    nps = np.linspace(0, 0.65, 66)
elif L>129:
    nps = np.linspace(0.22, 0.30, 5)
else:
    nps = np.around(np.linspace(0, 0.65, 33), 2)
for pz in nps:
    ps.append((pz, 0.35))
       

QFI = []       
for pz,pu in ps:
    psi = stim.TableauSimulator(seed=seed)
    psi.set_num_qubits(L)
    
    for _ in range(t):
        triphase_layer(psi, Us, pz, pu)

    C = corr_matrix(psi)
    file_name = f'L{L}_pz'+f'{pz:.3f}'[2:]+f'_pu'+f'{pu:.3f}'[2:]+f'_seed{seed}'
    qfi_traj, err_traj = sim_annealing(schedule, repeats, C, L)
    QFI.append([pz, pu, qfi_traj, err_traj])
if np.shape(QFI)[1] > 0:
    np.savetxt(f'triphase_data/QFI/pu350_L{L}_seed{seed}_qfi_ann.txt', QFI)
print(f'QFI density calculated for L {L} seed {seed}')
