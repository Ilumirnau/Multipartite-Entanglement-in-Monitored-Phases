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
    

#----------------------ENTANGLEMENT ENTROPY CALCULATION---------------------------------------
# all of them can be sped up with numba?
def binaryMatrix(zStabilizers):
    """
        - Purpose: Construct the binary matrix representing the stabilizer states.
        - Inputs:
            - zStabilizers (array): The result of conjugating the Z generators on the initial state.
        Outputs:
            - binaryMatrix (array of size (N, 2N)): An array that describes the location of the stabilizers in the tableau representation.
    """
    N = len(zStabilizers)
    binaryMatrix = np.zeros((N,2*N), dtype=int)
    r = 0 # Row number
    for row in zStabilizers:
        c = 0 # Column number
        for i in row:
            if i == 3: # Pauli Z
                binaryMatrix[r,N + c] = 1
            if i == 2: # Pauli Y
                binaryMatrix[r,N + c] = 1
                binaryMatrix[r,c] = 1
            if i == 1: # Pauli X
                binaryMatrix[r,c] = 1
            c += 1
        r += 1

    return binaryMatrix

def getCutStabilizers(binaryMatrix, cut):
    """
        - Purpose: Return only the part of the binary matrix that corresponds to the qubits we want to consider for a bipartition.
        - Inputs:
            - binaryMatrix (array of size (N, 2N)): The binary matrix for the stabilizer generators.
            - cut: location (int) or type (string) of cut
        - Outputs:
            - cutMatrix (array of size (N, 2cut)): The binary matrix for the cut on the left.
    """
    
    N = len(binaryMatrix)
    if type(cut)==type(1):
        cutMatrix = np.zeros((N,2*cut))

        cutMatrix[:,:cut] = binaryMatrix[:,:cut]
        cutMatrix[:,cut:] = binaryMatrix[:,N:N+cut]
    
    elif type(cut)==type('cut is a string'):
        if 'A' in cut:
            if 'B' in cut:
                subN = N//2
                cutMatrix = np.zeros((N,2*subN))
                #x in ab
                cutMatrix[:,:subN] = binaryMatrix[:,:subN]
                #z in ab
                cutMatrix[:,subN:] = binaryMatrix[:,N:N+subN]

            elif 'C' in cut:
                subN = N//2
                cutMatrix = np.zeros((N,2*subN))

                #x in a
                cutMatrix[:,:subN//2] = binaryMatrix[:,:subN//2]
                #z in a
                cutMatrix[:,subN:3*subN//2] = binaryMatrix[:,N:N+subN//2]
                #x in c
                cutMatrix[:,subN//2:subN] = binaryMatrix[:,subN:3*subN//2]
                #z in c
                cutMatrix[:,3*subN//2:] = binaryMatrix[:,N+subN:N+3*subN//2]
            else:
                subN = N//4
                cutMatrix = np.zeros((N,2*subN))
                #x in a
                cutMatrix[:,:subN] = binaryMatrix[:,:subN]
                #z in a
                cutMatrix[:,subN:] = binaryMatrix[:,N:N+subN]
        elif 'B' in cut:
            if 'C' in cut:
                subN = N//2
                cutMatrix = np.zeros((N,2*subN))

                #x in bc
                cutMatrix[:,:subN] = binaryMatrix[:,subN//2:3*subN//2]
                #z in bc
                cutMatrix[:,subN:] = binaryMatrix[:,N+subN//2:N+3*subN//2]

            else:
                subN = N//4
                cutMatrix = np.zeros((N,2*subN))
                #x in b
                cutMatrix[:,:subN] = binaryMatrix[:,subN:2*subN]
                #z in b
                cutMatrix[:,subN:] = binaryMatrix[:,N+subN:N+2*subN]
        elif 'C' in cut:
            subN = N//4
            cutMatrix = np.zeros((N,2*subN))

            #x in c
            cutMatrix[:,:subN] = binaryMatrix[:,2*subN:3*subN]
            #z in c
            cutMatrix[:,subN:] = binaryMatrix[:,N+2*subN:N+3*subN]            

        else:
            subN = N//4
            cutMatrix = np.zeros((N,2*subN))

            #x in d
            cutMatrix[:,:subN] = binaryMatrix[:,3*subN:N]
            #z in d
            cutMatrix[:,subN:] = binaryMatrix[:,N+3*subN:]       

    return cutMatrix

def gf2_rank(A):
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


def ee(psi: stim.TableauSimulator, cut):
    '''
    This function calculates the entanglement entropy of the circuit given a cut.
    psi: Tableau state
    cut: number of qubits of the subsystem whose ent. entropy is being calculated (N_A) or type as string for TMI
    '''
    # Create the tableau representation
    tableau = psi.current_inverse_tableau() ** -1
    zs = [tableau.z_output(k) for k in range(len(tableau))]
    zs = np.array(zs)

    # Cut the binary matrix that corresponds to the qubits we want to consider for a bipartition
    binMat = binaryMatrix(zs)
    cutMatrix = getCutStabilizers(binMat, cut)

    # Calculate the rank of the projected matrix via Gaussian elimination
    rank = gf2_rank(cutMatrix)

    N = psi.num_qubits
    # Calculate the entanglement entropy: S = rank(cutMatrix) - cut
    if type(cut)==type(0):
        cut_size = cut
    else:
        cut_size = N * len(cut) // 4

    S = rank - cut_size

    return S

# TOPOLOGICAL ENTANGLEMENT ENTROPY

# use numba to vectorize?
def S_topo(psi: stim.TableauSimulator):
    '''
    Returns the topological entanglement entropy, the system's partition is ABDC (notice order)
    Stopo=Sab+Sbc-Sb-Sabc
    We name variables as in formula, but the function ee() is defined as ABCD, not ABDC
    '''
    sb = ee(psi, 'B')
    sab = ee(psi, 'AB')
    sbc = ee(psi, 'AC')
    sabc = ee(psi, 'C')

    return sab + sbc - sb - sabc

def tmi(psi: stim.TableauSimulator):
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
    if L <= 33:
        C = corr_matrix(psi)
        file_name = f'L{L}_pz'+f'{pz:.3f}'[2:] + \
            '_pu' + f'{pu:.3f}'[2:]+f'_seed{seed}'
        np.save(f'data/correlations/{file_name}_C_', C)
np.savetxt(f'data/surface_entanglement/L{L}_seed{seed}_S_.txt',
           np.array(ent_data))
