# Script to use KDM/FDM Krylov subspace methods to
# compute ground state energies
# This script does fragment QKSDs using LAS subspaces
# followed by a whole-system QKSD, to obtain energies

import numpy as np
import scipy.linalg as LA
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, save_npz, load_npz, linalg

from argparse import ArgumentParser

# Qiskit imports
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.circuit.library.initial_states import HartreeFock
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy, ElectronicStructureDriverResult
from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import PauliTrotterEvolution, MatrixEvolution
from qiskit.opflow.state_fns import CircuitStateFn, StateFn
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
from qiskit import Aer, transpile, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance

# PySCF imports
from pyscf import gto, lib, scf, mcscf, ao2mo

# MRH imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def get_scipy_csc_from_op(Hop):
    return csc_matrix(Hop, dtype=complex)

def apply_time_evolution_op(statevector, Hcsc, dt, nstates):
    return expm_multiply(-1j*Hcsc*dt, statevector, start=0.0, stop=nstates-1, num=nstates)
    
# The number of timesteps and step size
# as arguments
parser = ArgumentParser(description='Do a LAS-QKSD (KDM H) using matrix forms.')
parser.add_argument('--steps', type=int, default=5, help='Number of time steps for the Krylov subspace')
parser.add_argument('--tau', type=float, default=0.1, help='Step size of the time steps')
args = parser.parse_args()

time_steps=args.steps
tau=args.tau

#xyz = '''H 0.0 0.0 0.0
#         H 1.0 0.0 0.0
#         H 2.5 0.0 0.0
#         H 3.5 0.0 0.0
#         H 5.0 0.0 0.0
#         H 6.0 0.0 0.0'''
xyz = '''H 0.0 0.0 0.0
         H  0.0 0.0 0.6
         H  0.0 0.0 1.5
         H  0.0 0.0 2.0'''

# PySCF RHF for CAS
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log',
             symmetry=False)

# Do RHF
mf = scf.RHF(mol).run()

# The LASSCF here is needed so the orbital space is the localized
# active space
las = LASSCF(mf, (2,2),(2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(loc_mo_coeff)
loc_mo_coeff = las.mo_coeff
print("LASSCF energy: ", las.e_tot)
nuc_rep_en = las.energy_nuc()
print("Nuclear repulsion: ", las.energy_nuc())
ncas_sub = las.ncas_sub
nelecas_sub = las.nelecas_sub

## Fragment QKSD
## Use QKSD together with the LAS fragment Hamiltonians
## to give better fragment wavefunctions

h1_las = las.h1e_for_cas()
eri_las = las.get_h2eff(loc_mo_coeff)

# Storing each fragment's h1 and h2 as a list
h1_frag = []
h2_frag = []

for idx in range(len(ncas_sub)):
    h1_frag.append(h1_las[idx][0][0])
    h2_frag.append(las.get_h2eff_slice(eri_las, idx))

# Checking that the fragment Hamiltonian shapes are correct
for f in range(len(ncas_sub)):
    print("H1_frag shape: ", h1_frag[f].shape)
    print("H2_frag shape: ", h2_frag[f].shape)

coeff_list = []
wfn_list = []
ham_list = []

for frag in range(len(ncas_sub)):
    n_so = 2*ncas_sub[frag]
    n_alpha = nelecas_sub[frag][0]
    n_beta = nelecas_sub[frag][1]

    particle_number = ParticleNumber(
        num_spin_orbitals=n_so,
        num_particles=(n_alpha, n_beta),
    )
    electronic_energy = ElectronicEnergy.from_raw_integrals(
            # Using MO basis here for simplified conversion
            ElectronicBasis.MO, h1_frag[frag], h2_frag[frag]
    )
    driver_result = ElectronicStructureDriverResult()
    driver_result.add_property(electronic_energy)
    driver_result.add_property(particle_number)

    second_q_ops = driver_result.second_q_ops()

    # Choose fermion-to-qubit mapping
    qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
    # This just outputs a qubit op corresponding to a 2nd quantized op
    qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
    hamiltonian = qubit_ops[0]

    # Numpy solver to estimate error
    np_solver = NumPyEigensolver(k=1)
    ed_result = np_solver.compute_eigenvalues(hamiltonian)
    np_en = ed_result.eigenvalues
    print("NumPy result: ", np_en)
    #numpy_wfn = ed_result.eigenstates

    # Create a unitary by exponentiating the Hamiltonian
    # Using the scipy sparse matrix form
    ham_mat = hamiltonian.to_matrix()
    Hsp = get_scipy_csc_from_op(ham_mat)
    save_npz(f"h4_hsp_frag_{frag}.npz", Hsp)

    # Create a Hartree-Fock state
    init_state = HartreeFock(n_so, (n_alpha, n_beta), qubit_converter)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(init_state, backend=backend, shots=1, memory=True)
    job_result = job.result()
    init_statevector = np.asarray(job_result.get_statevector(init_state)._data, dtype=complex)

    # Initialize F, S matrices
    F_mat = np.zeros((time_steps, time_steps), dtype=complex)
    S_mat = np.zeros((time_steps, time_steps), dtype=complex)

    # U |\phi_0>
    statevector = apply_time_evolution_op(init_statevector, Hsp, tau, time_steps)
    omega_list = [np.asarray(state, dtype=complex) for state in statevector]
    wfn_list.append(omega_list)
    
    print("Norm of Krylov basis states: ",np.linalg.norm(omega_list[1]))

    # V U |\phi_0>
    Homega_list = [Hsp.dot(omega) for omega in omega_list]
    ham_list.append(Homega_list)

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
        #print(F_mat)

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

        idx = eigvals.argsort()
        eigvals = eigvals[idx]
        print("Eigvals: ", eigvals)
        eigvecs = eigvecs[:,idx]
        #print("Eigvecs: ", eigvecs)
    # QKSD Wavefunction
    qksd_wfn = []
    c_list = []
    for m in range(time_steps):
        c_list.append(eigvecs[m][0])
    print("QKSD wfn coefficients: ", c_list)
    print("QKSD wfn: ", np.sum([np.dot(eigvecs[m][0], omega_list[m]) for m in range(time_steps)]))
    coeff_list.append(c_list)

print("Norms: ")
print([np.linalg.norm(wfn_list[0][x]) for x in range(time_steps)])

## Whole-system QKSD

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
    print("Exact diagonalization result: ", eigvals)

except:
    # If not stored, create a qubit Hamiltonian
    # To obtain the qubit Hamiltonian

    particle_number = ParticleNumber(
        num_spin_orbitals=n_so,
        num_particles=(n_alpha, n_beta),
    )

    # Assuming an RHF reference for now, so h1_b, h2_ab, h2_bb are created using
    # the corresponding spots from h1_frag and just the aa term from h2_frag
    electronic_energy = ElectronicEnergy.from_raw_integrals(
            # Using MO basis here for simplified conversion
            ElectronicBasis.MO, h1, h2)

    driver_result = ElectronicStructureDriverResult()
    driver_result.add_property(electronic_energy)
    driver_result.add_property(particle_number)

    second_q_ops = driver_result.second_q_ops()
    qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
    qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
    hamiltonian = qubit_ops[0]
    #print(hamiltonian)

    # Numpy solver to estimate error
    np_solver = NumPyEigensolver(k=1)
    ed_result = np_solver.compute_eigenvalues(hamiltonian)
    np_en = ed_result.eigenvalues
    print("NumPy result: ", np_en)
    #numpy_wfn = ed_result.eigenstates

    # Create a unitary by exponentiating the Hamiltonian
    # Using the scipy sparse matrix form
    ham_mat = hamiltonian.to_matrix()
    Hsp = get_scipy_csc_from_op(ham_mat)
    save_npz("h4_hsp_las.npz", Hsp)

en_list = []
tot_coeffs_list = []
F_tot = np.zeros((time_steps, time_steps), dtype=complex)
S_tot = np.zeros((time_steps, time_steps), dtype=complex)
for step1 in range(time_steps):
    for step2 in range(time_steps):
        frag0_state = wfn_list[0][step1]
        frag1_state = wfn_list[1][step2]

        init_statevector = np.kron(frag0_state, frag1_state) 

        # Initialize F, S matrices
        F_mat = np.zeros((time_steps, time_steps), dtype=complex)
        S_mat = np.zeros((time_steps, time_steps), dtype=complex)

        # U |\phi_0>
        statevector = apply_time_evolution_op(init_statevector, Hsp, tau, time_steps)
        omega_list = [np.asarray(coeff_list[0][step1]*coeff_list[1][step2]*state, dtype=complex) for state in statevector]

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
            #print(S_mat)
        F_tot += F_mat
        S_tot += S_mat

for m in range(time_steps):
    # Using an SVD to condition the F matrix
    # Before doing the eigendecomposition
    Stol=1e-12
    U, s, Vh = LA.svd(S_tot[:m+1, :m+1])

    #print(np.allclose(U, Vh.T.conj()))
    Dtemp = 1/np.sqrt(s)
    Dtemp[Dtemp**2 > 1/Stol] = 0

    Xp = U[0:len(s),0:len(Dtemp)]*Dtemp
    Fp = Xp.T.conjugate() @ F_tot[:m+1,:m+1] @ Xp

    # Eigenvalues of the conditioned matrix
    eigvals, eigvecs = LA.eig(Fp)

    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    print("Eigvals: ", eigvals)
    eigvecs = eigvecs[:,idx]
    print("QKSD energy",m+1,"  = ", eigvals[0])
    tot_coeffs = eigvecs[:, 0]

print("Total energy: ", eigvals[0])
