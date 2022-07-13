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

# PySCF imports
from pyscf import gto, scf

# MRH imports
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def get_scipy_csc_from_op(Hop, factor):
    return csc_matrix(factor*Hop, dtype=complex)

def apply_time_evolution_op(statevector, Hcsc, dt, nstates):
    return expm_multiply(Hcsc, statevector, start=0.0, stop=dt*nstates, num=nstates, endpoint=False)
    
def get_so_ci_vec(ci_vec, nsporbs,nelec):
    lookup = {}
    cnt = 0
    norbs = nsporbs//2

    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == np.sum(nelec)//2:
            lookup[f"{ii:0{norbs}b}"] = cnt
            cnt +=1
    # This is just indexing the hilber space from 0,1,...,mCn
    #print (lookup)

    so_ci_vec = np.zeros(2**nsporbs)
    for kk in range (2**nsporbs):
        if f"{kk:0{nsporbs}b}"[norbs:].count('1')==nelec[0] and f"{kk:0{nsporbs}b}"[:norbs].count('1')==nelec[1]:
            so_ci_vec[kk] = ci_vec[lookup[f"{kk:0{nsporbs}b}"[norbs:]],lookup[f"{kk:0{nsporbs}b}"[:norbs]]]

    return so_ci_vec


xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 2.5 0.0 0.0
         H 3.5 0.0 0.0
         H 5.0 0.0 0.0
         H 6.0 0.0 0.0'''
#xyz = '''H 0.0 0.0 0.0
#         H  0.0 0.0 1.5
#         H  0.0 0.0 3.0
#         H  0.0 0.0 4.5'''

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
#print(hamiltonian)

# Numpy solver to estimate error
np_solver = NumPyEigensolver(k=1)
ed_result = np_solver.compute_eigenvalues(hamiltonian)
np_en = ed_result.eigenvalues
print("NumPy result: ", np_en+nuc_rep_en)
#numpy_wfn = ed_result.eigenstates

# Set up the number of timesteps and step size
time_steps=6
tau=0.1

# Initialize F, S matrices
F_mat = np.zeros((time_steps, time_steps), dtype=complex)
S_mat = np.zeros((time_steps, time_steps), dtype=complex)

# Create a unitary by exponentiating the Hamiltonian
# Using the scipy sparse matrix form
ham_mat = hamiltonian.to_matrix()
Hsp = get_scipy_csc_from_op(ham_mat, -1.0j)

'''
# Create a Hartree-Fock state
init_state = HartreeFock(n_so, (n_alpha, n_beta), qubit_converter)
backend = Aer.get_backend('statevector_simulator')
job = execute(init_state, backend=backend, shots=1, memory=True)
job_result = job.result()
init_statevector = np.asarray(job_result.get_statevector(init_state)._data, dtype=complex)
'''

# Create a LAS wave function and initialize
# PySCF RHF for LAS
mol = gto.M (atom = xyz, basis = 'sto-3g', output='h6_sto3g.log',
             symmetry=False)

# Do RHF
mf = scf.RHF(mol).run()
las = LASSCF(mf, (2,2,2),(2,2,2), spin_sub=(1,1,1))
frag_atom_list = ((0,1),(2,3),(4,5))
loc_mo_coeff = las.localize_init_guess(frag_atom_list, mf.mo_coeff)
las.kernel(loc_mo_coeff)
print("LASSCF energy: ", las.e_tot)

qr1 = QuantumRegister(np.sum(las.ncas_sub)*2, 'q1')
new_circuit = QuantumCircuit(qr1)
new_circuit.initialize(get_so_ci_vec(las.ci[0][0],2*las.ncas_sub[0],las.nelecas_sub[0]), [0,1,6,7])
new_circuit.initialize(get_so_ci_vec(las.ci[1][0],2*las.ncas_sub[1],las.nelecas_sub[1]), [2,3,8,9])
new_circuit.initialize(get_so_ci_vec(las.ci[2][0],2*las.ncas_sub[2],las.nelecas_sub[2]), [4,5,10,11])

backend = Aer.get_backend('statevector_simulator')
job = execute(new_circuit, backend=backend, shots=1, memory=True)
job_result = job.result()
init_statevector = np.asarray(job_result.get_statevector(new_circuit)._data, dtype=complex)

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
Homega_list = [np.dot(ham_mat, omega) for omega in omega_list]

for m in range(time_steps):
    # Filling the S matrix
    for n in range(m+1):
        # < \phi_0 | U_m^+ U_n | \phi_0 >
        Smat_el = np.vdot(omega_list[m], omega_list[n])

        print("S_{}_{} = {}".format(m, n, Smat_el))
        S_mat[m][n] = Smat_el
        S_mat[n][m] = np.conj(Smat_el)

    # Filling the F matrix
    # < \phi_0 | U_m^+ V U_n | \phi_0 >
    for n in range(m+1):
        Fmat_el = np.vdot(omega_list[m], Homega_list[n])
        print("F_{}_{} = {}".format(m, n, Fmat_el))
        F_mat[m][n] = Fmat_el
        F_mat[n][m] = np.conj(Fmat_el)
print(S_mat)

# Using the generalized Schur decomposition of F, S
# to obtain the eigvals via scipy.linalg.ordqz (ordered QZ)
AA, BB, alpha, beta, Q, Z = LA.ordqz(F_mat, S_mat, sort='lhp')
#AA, BB, Q, Z = LA.qz(F_mat, S_mat)

alpha = np.diag(AA)
beta = np.diag(BB)
print("Alpha: ",alpha)
print("Beta: ",beta)
eigvals = alpha/beta
print("eigvals = ", np.sort(eigvals))
print("QKSD energy = ", eigvals[0]+nuc_rep_en)

print("Error for timesteps={}: {}".format(time_steps, eigvals[0]-np_en))
