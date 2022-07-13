# Script to use KDM/FDM Krylov subspace methods to
# compute ground state energies
# Currently only computes KDM H

import numpy as np
import scipy.linalg as LA
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix

# Qiskit imports
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.circuit.library.initial_states import HartreeFock
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import PauliTrotterEvolution, MatrixEvolution
from qiskit.opflow.state_fns import CircuitStateFn, StateFn
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
from qiskit import Aer, transpile, execute
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance

def get_scipy_csc_from_op(Hop, factor):
    return csc_matrix(factor*Hop, dtype=complex)

def apply_time_evolution_op(statevector, Hcsc, dt, nstates):

    qc_vec = np.array(statevector, dtype=complex)

    return expm_multiply(Hcsc, qc_vec, start=0.0, stop=dt*nstates, num=nstates, endpoint=True)
    
#xyz = '''H 0.0 0.0 0.0
#         H 1.5 0.0 0.0
#         H 3.0 0.0 0.0
#         H 4.5 0.0 0.0
#         H 6.0 0.0 0.0
#         H 7.5 0.0 0.0'''
xyz = '''H 0.0 0.0 0.0
         H  0.0 0.0 1.5
         H  0.0 0.0 3.0
         H  0.0 0.0 4.5'''

# First, perform an RHF calculation using the qiskit_nature PySCF driver
driver = PySCFDriver(atom=xyz, charge=0, spin=0, method=MethodType.RHF)
driver_result = driver.run()
electronic_en = driver_result.get_property(ElectronicEnergy)
nuc_rep_en = electronic_en.nuclear_repulsion_energy

# Save number of orbitals, alpha and beta electrons from driver result
part_num = driver_result.get_property(ParticleNumber)
n_so = part_num.num_spin_orbitals
n_alpha = part_num.num_alpha
n_beta = part_num.num_beta

# Extract and convert the 1 and 2e integrals
# To obtain the qubit Hamiltonian
second_q_ops = driver_result.second_q_ops()
qubit_converter = QubitConverter(mapper = JordanWignerMapper(), two_qubit_reduction=False)
qubit_ops = [qubit_converter.convert(op) for op in second_q_ops]
hamiltonian = qubit_ops[0]
print(hamiltonian)

# Numpy solver to estimate error
np_solver = NumPyEigensolver(k=1)
ed_result = np_solver.compute_eigenvalues(hamiltonian)
np_en = ed_result.eigenvalues
print("NumPy result: ", np_en+nuc_rep_en)
#numpy_wfn = ed_result.eigenstates

# Set up the number of timesteps and step size
time_steps=4
tau=0.1

# Initialize F, S matrices
F_mat = np.zeros((time_steps, time_steps), dtype=complex)
S_mat = np.zeros((time_steps, time_steps), dtype=complex)
#np.fill_diagonal(S_mat, 1.0)

# Create a unitary by exponentiating the Hamiltonian
# And trotterizing onto the circuit
ham_mat = hamiltonian.to_matrix()
Hsp = get_scipy_csc_from_op(ham_mat, -1.0j)

# Create a Hartree-Fock state
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

omega_list = []
Homega_list = []

statevector = apply_time_evolution_op(init_statevector, Hsp, tau, time_steps)

for m in range(time_steps):
    # U |\phi_0>
    omega_list.append(np.asarray(statevector[m], dtype=complex))
    
    # Filling the S matrix
    for n in range(len(omega_list)):
        # < \phi_0 | U | \phi_0 >
        Smat_el = np.vdot(omega_list[m], omega_list[n])

        print("S_{}_{} = {}".format(m, n, Smat_el))
        S_mat[m][n] = Smat_el
        S_mat[n][m] = np.conj(Smat_el)

    # V U |\phi_0>
    h_unitary = hamiltonian.to_matrix()
    h_statevector = np.dot(h_unitary, omega_list[m])
    Homega_list.append(np.asarray(h_statevector, dtype=complex))

    # Filling the F matrix
    # < \phi_0 | U^+ V U | \phi_0 >
    for n in range(len(omega_list)):
        Fmat_el = np.vdot(omega_list[m], Homega_list[n])
        print("F_{}_{} = {}".format(m, n, Fmat_el))
        F_mat[m][n] = Fmat_el
        F_mat[n][m] = np.conj(Fmat_el)
print(omega_list)
print(S_mat)

# Using the generalized Schur decomposition of F, S
# to obtain the eigvals via scipy.linalg.ordqz (ordered QZ)
#AA, BB, alpha, beta, Q, Z = LA.ordqz(F_mat, S_mat, sort='lhp')
AA, BB, Q, Z = LA.qz(F_mat, S_mat)

alpha = np.diag(AA)
beta = np.diag(BB)
print("Alpha: ",alpha)
print("Beta: ",beta)
eigvals = alpha/beta
print("eigvals = ", np.sort(eigvals))
print("QKSD energy = ", eigvals[0]+nuc_rep_en)

print("Error for timesteps={}: {}".format(time_steps, eigvals[0]-np_en))
