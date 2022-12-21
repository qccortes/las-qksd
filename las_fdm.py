# Script to use FDM Krylov subspace methods to
# compute ground state energies
# Using the converged LAS energies of each fragment
# as filter energies

import numpy as np
import scipy.linalg as LA
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, save_npz, load_npz, linalg

from argparse import ArgumentParser

# Qiskit imports
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter
from qiskit_nature.second_q.circuit.library.initial_states import HartreeFock
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit.algorithms import NumPyEigensolver
from qiskit import Aer, transpile, execute

# PySCF imports
from pyscf import gto, lib, scf, mcscf, ao2mo

# MRH imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def get_scipy_csc_from_op(Hop):
    return csc_matrix(Hop, dtype=complex)

def apply_time_evolution_op(statevector, Hcsc, dt, nstates):
    return expm_multiply(-1j*Hcsc*dt, statevector, start=0.0, stop=nstates-1, num=nstates)
    
## This function makes a few assumptions
## 1. The civector is arranged as a 2D matrix of coeffs
##    of size [nalphastr, nbetastr]
## 2. The civector contains all configurations within
##    the (localized) active space
def get_so_ci_vec(ci_vec, nsporbs,nelec):
    lookup_a = {}
    lookup_b = {}
    cnt = 0
    norbs = nsporbs//2

    # Here, we set up a lookup dictionary which is
    # populated when either the number of alpha e-s
    # or the number of beta electrons is correct
    # It stores "bitstring" : decimal_value pairs
    ## The assumption is that nalpha==nbeta
    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == nelec[0]:
            lookup_a[f"{ii:0{norbs}b}"] = cnt
            cnt +=1

    cnt = 0
    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == nelec[1]:
            lookup_b[f"{ii:0{norbs}b}"] = cnt
            cnt +=1
    # This is just indexing the hilber space from 0,1,...,mCn
    #print (lookup)

    # Here the spin orbital CI vector is populated
    # the same lookup is used for alpha and beta, but for two different
    # sections of the bitstring
    so_ci_vec = np.zeros(2**nsporbs)
    for kk in range (2**nsporbs):
        if f"{kk:0{nsporbs}b}"[norbs:].count('1')==nelec[0] and f"{kk:0{nsporbs}b}"[:norbs].count('1')==nelec[1]:
            so_ci_vec[kk] = ci_vec[lookup_a[f"{kk:0{nsporbs}b}"[norbs:]],lookup_b[f"{kk:0{nsporbs}b}"[:norbs]]]

    return so_ci_vec

# The number of timesteps and step size
# as arguments
parser = ArgumentParser(description='Do a LAS-QKSD (FDM H) using matrix forms.')
parser.add_argument('--fsteps', type=int, default=5, help='Number of time steps for the Krylov subspace of each fragment')
parser.add_argument('--steps', type=int, default=5, help='Number of time steps for the Krylov subspace')
parser.add_argument('--ftau', type=float, default=0.1, help='Step size of the time steps for fragments')
parser.add_argument('--tau', type=float, default=0.1, help='Step size of the time steps')
args = parser.parse_args()

time_steps=args.steps
ftime_steps = args.fsteps
tau=args.tau
ftau = args.ftau

#xyz = '''H 0.0 0.0 0.0
#         H 1.0 0.0 0.0
#         H 2.5 0.0 0.0
#         H 3.5 0.0 0.0
#         H 5.0 0.0 0.0
#         H 6.0 0.0 0.0'''
xyz = '''H 0.0 0.0 0.0
         H  0.0 0.0 0.5
         H  0.0 0.0 1.5
         H  0.0 0.0 2.0'''

# Create a LAS wave function and initialize
# PySCF RHF for LAS
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
             symmetry=False, verbose=10)

# Do RHF
mf = scf.RHF(mol).run()
las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
e_states = las.kernel(loc_mo_coeff)[1]
loc_mo_coeff = las.mo_coeff
print(e_states)
print("LASSCF energy: ", las.e_tot)
print("Subspace energies: ", e_states)
nuc_rep_en = las.energy_nuc()
ncas_sub = las.ncas_sub

# Create fragment integrals
h1_las = las.h1e_for_cas(loc_mo_coeff)
h2_las = las.get_h2eff(loc_mo_coeff)
#
## Storing each fragment's h1 and h2 as a list
h1_frag = []
h2_frag = []

for idx in range(len(ncas_sub)):
    h1_frag.append(h1_las[idx][0][0])
    h2_frag.append(las.get_h2eff_slice(h2_las, idx))

frag_omega_list = []
for frag in range(len(ncas_sub)):
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1_frag[frag], h2_frag[frag])
    second_q_op = electronic_energy.second_q_op()
    #print(second_q_op)
    qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
    ham_frag = qubit_converter.convert(second_q_op)
    #print(hamiltonian)
    ham_mat = ham_frag.to_matrix()
    Hsp = get_scipy_csc_from_op(ham_mat)
    save_npz(f"h4_hsp_frag{frag}.npz", Hsp)

    # Create a Hartree-Fock state |HF>
    n_so = las.ncas_sub[frag]
    (n_alpha, n_beta) = tuple(las.nelecas_sub[frag])
    init_state = HartreeFock(n_so, (n_alpha, n_beta), qubit_converter)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(init_state, backend=backend, shots=1, memory=True)
    job_result = job.result()
    init_statevector = np.asarray(job_result.get_statevector(init_state)._data, dtype=complex)
    # U_frag |HF>
    statevector = apply_time_evolution_op(init_statevector, Hsp, ftau, ftime_steps)
    omega_list = [np.asarray(state, dtype=complex) for state in statevector]
    frag_omega_list.append(omega_list)

## Block to test if the np.eig of the fragment Hamiltonians
## gives the same answer as the subspace CI energies
#    # Numpy solver to estimate error
#    np_solver = NumPyEigensolver(k=1)
#    ed_result = np_solver.compute_eigenvalues(hamiltonian)
#    np_en = ed_result.eigenvalues
#    print("NumPy result: ", np_en)
#    #numpy_wfn = ed_result.eigenstates
#exit()

# Extract and convert the 1 and 2e integrals
mc = mcscf.CASCI(mf,4,4)
h1, e_core = mc.h1e_for_cas(mo_coeff=loc_mo_coeff)
print("Core energy: ", e_core)
h2 = ao2mo.restore(1, mc.get_h2eff(loc_mo_coeff), mc.ncas)

# Save number of orbitals, alpha and beta electrons from driver result
n_so = 2*mc.ncas
n_alpha = mc.nelecas[0]
n_beta = mc.nelecas[1]

# Extract and convert the 1 and 2e integrals
# To obtain the qubit Hamiltonian

# Check if hsp.npy exists:
try:
    Hsp = load_npz("h4_hsp_las.npz")
    print("Successfully loaded Hamiltonian.")
    # Diagonalization for reference
    eigvals, eigvecs = linalg.eigs(Hsp, k=1)
    print("Exact diagonalization result: ", eigvals-e_core)

except:
    # If not stored, create a qubit Hamiltonian
    # To obtain the qubit Hamiltonian

    #particle_number = ParticleNumber(
    #    num_spin_orbitals=n_so,
    #    num_particles=(n_alpha, n_beta),
    #)

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)

    #problem = ElectronicStructureProblem(electronic_energy)
    second_q_op = electronic_energy.second_q_op()
    qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
    hamiltonian = qubit_converter.convert(second_q_op)

    # Numpy solver to estimate error
    np_solver = NumPyEigensolver(k=1)
    ed_result = np_solver.compute_eigenvalues(hamiltonian)
    np_en = ed_result.eigenvalues
    print("NumPy result: ", np_en-e_core)
    #numpy_wfn = ed_result.eigenstates

    # Create a unitary by exponentiating the Hamiltonian
    # Using the scipy sparse matrix form
    ham_mat = hamiltonian.to_matrix()
    Hsp = get_scipy_csc_from_op(ham_mat)
    save_npz("h4_hsp_las.npz", Hsp)

# Initialize F, S matrices
F_tot = np.zeros((time_steps, time_steps), dtype=complex)
S_tot = np.zeros((time_steps, time_steps), dtype=complex)

for fstep in range(ftime_steps):
    for frag in range(len(ncas_sub)):
        if frag == 0:
            prev_omega = frag_omega_list[frag][fstep]
        else:
            frag_omega = frag_omega_list[frag][fstep]
            prev_omega = np.kron(frag_omega, prev_omega)
    init_statevector = prev_omega

    # Initialize F, S matrices
    F_mat = np.zeros((time_steps, time_steps), dtype=complex)
    S_mat = np.zeros((time_steps, time_steps), dtype=complex)

    # U |\phi_0>
    statevector = apply_time_evolution_op(init_statevector, Hsp, tau, time_steps)
    omega_list = [np.asarray(state, dtype=complex) for state in statevector]

    # V U |\phi_0>
    Homega_list = [Hsp.dot(omega) for omega in omega_list]

    for m in range(time_steps):
        # Filling the S matrix
        for n in range(m+1):
            # < \phi_0 | U_m^+ U_n | \phi_0 >
            Smat_el = np.vdot(omega_list[m], omega_list[n])

            #print("S_{}_{} = {}".format(m, n, Smat_el))
            S_mat[m][n] = Smat_el
            S_mat[n][m] = np.conj(Smat_el)

        # Filling the F matrix
        # < \phi_0 | U_m^+ V U_n | \phi_0 >
        for n in range(m+1):
            Fmat_el = np.vdot(omega_list[m], Homega_list[n])
            #print("F_{}_{} = {}".format(m, n, Fmat_el))
            F_mat[m][n] = Fmat_el
            F_mat[n][m] = np.conj(Fmat_el)
    F_tot += F_mat
    S_tot += S_mat

en_list = []
for m in range(time_steps):
    # Using an SVD to condition the F matrix
    # Before doing the eigendecomposition
    Stol=1e-12
    U, s, Vh = LA.svd(S_mat[:m+1, :m+1])

    #print(np.allclose(U, Vh.T.conj()))
    Dtemp = 1/np.sqrt(s)
    Dtemp[Dtemp**2 > 1/Stol] = 0

    Xp = U[0:len(s),0:len(Dtemp)]*Dtemp
    Fp = Xp.T.conjugate() @ F_mat[:m+1,:m+1] @ Xp

    # Eigenvalues of the conditioned matrix
    eigvals, eigvecs = LA.eig(Fp)
    #print("eigvals = ", np.sort(eigvals))
    print("QKSD energy = ", np.real(eigvals[0])-e_core)
    en_list.append(np.real(eigvals[0])-e_core)

print(en_list)
