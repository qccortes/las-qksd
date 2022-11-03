# Script to use KDM/FDM Krylov subspace methods to
# compute ground state energies
# Currently only computes KDM H

import numpy as np
import scipy.linalg as LA
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, save_npz, load_npz, linalg

from argparse import ArgumentParser

# PySCF imports
from pyscf import gto, lib, scf, mcscf, ao2mo

# Qiskit imports
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.drivers import UnitsType
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

def get_scipy_csc_from_op(Hop):
    return csc_matrix(Hop, dtype=complex)

def apply_time_evolution_op(statevector, Hcsc, dt, nstates):
    return expm_multiply(-1j*Hcsc*dt, statevector, start=0.0, stop=nstates-1, num=nstates)
    
# The number of timesteps and step size
# as arguments
parser = ArgumentParser(description='Do a QKSD (KDM H) using matrix forms.')
parser.add_argument('--steps', type=int, default=5, help='Number of time steps for the Krylov subspace')
parser.add_argument('--tau', type=float, default=0.1, help='Step size of the time steps')
args = parser.parse_args()

time_steps=args.steps
tau=args.tau

xyz = '''H 0.0 0.0 0.0
         H  0.0 0.0 0.5
         H  0.0 0.0 1.5
         H  0.0 0.0 2.0'''
#xyz = '''H 0.0 0.0 0.0
#         H 0.0 0.0 0.5
#         H 0.0 0.0 1.0
#         H 0.0 0.0 1.5'''

# Perform an RHF calculation using PySCF
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h4_sto3g.log', charge=0, spin=0,
             symmetry=False, verbose=lib.logger.DEBUG)
mf = scf.RHF(mol).newton()
hf_en = mf.kernel()
print("HF energy: ",hf_en)
nuc_rep_en = mf.energy_nuc()

# Set up CASCI for active orbitals
mc = mcscf.CASCI(mf,4,(2,2))
n_so = 2*mc.ncas
(n_alpha, n_beta) = (mc.nelecas[0], mc.nelecas[1])

# Extract and convert the 1 and 2e integrals
# To obtain the qubit Hamiltonian
h1, e_core = mc.h1e_for_cas()
print("Core energy:",e_core)
h2 = ao2mo.restore(1, mc.get_h2eff(), mc.ncas)

# Check if hsp.npy exists:
try:
    Hsp = load_npz("h4_hsp.npz")
    print("Successfully loaded Hamiltonian.")
    # Diagonalization for reference
    eigvals, eigvecs = linalg.eigs(Hsp, k=1)
    print("Exact diagonalization result: ", eigvals)
except:
    # If not stored, create a qubit Hamiltonian
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
    save_npz("h4_hsp.npz", Hsp)
    #print("Hsp: \n",Hsp)

# Create a Hartree-Fock state
qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
init_state = HartreeFock(n_so, (n_alpha, n_beta), qubit_converter)
backend = Aer.get_backend('statevector_simulator')
job = execute(init_state, backend=backend, shots=1, memory=True)
job_result = job.result()
init_statevector = np.asarray(job_result.get_statevector(init_state)._data, dtype=complex)

'''
# Alternative way to obtain the HF statevector
bitstring = np.zeros(n_so)
bitstring[:n_alpha] = 1
bitstring[int(n_so/2):int(n_so/2)+n_beta] = 1
bitstring = np.flip(bitstring)
bitstring = ''.join(str(int(b)) for b in bitstring)
init_statevector = np.zeros(2**n_so)
init_statevector[int(bitstring, 2)] = 1
print(np.nonzero(init_statevector))
'''

# U |\phi_0>
statevector = apply_time_evolution_op(init_statevector, Hsp, tau, time_steps)
omega_list = [np.asarray(state, dtype=complex) for state in statevector]

# V U |\phi_0>
Homega_list = [Hsp.dot(omega) for omega in omega_list]

# Initialize F, S matrices
F_mat = np.zeros((time_steps, time_steps), dtype=complex)
S_mat = np.zeros((time_steps, time_steps), dtype=complex)

en_list = []
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

    #eigvals, eigvecs = LA.eig(F_mat[:m+1,:m+1], S_mat[:m+1,:m+1])
    
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

    '''
    # Using the generalized Schur decomposition of F, S
    # to obtain the eigvals via scipy.linalg.ordqz (ordered QZ)
    AA, BB, alpha, beta, Q, Z = LA.ordqz(F_mat[:m+1, :m+1], S_mat[:m+1, :m+1], sort='lhp')
    #AA, BB, Q, Z = LA.qz(F_mat, S_mat)

    #alpha = np.diag(AA)
    #beta = np.diag(BB)
    #print("Alpha: ",alpha)
    #print("Beta: ",beta)
    eigvals = alpha/beta
    '''
    #print("eigvals = ", np.sort(eigvals))
    print("QKSD energy = ", eigvals[0])
    en_list.append(np.real(eigvals[0]))

    #print("Error for timesteps={}: {}".format(m+1, eigvals[0]-np_en))
print([x+nuc_rep_en for x in en_list])
