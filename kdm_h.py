# Script to use KDM/FDM Krylov subspace methods to
# compute ground state energies
# Currently only computes KDM H

import numpy as np
import scipy.linalg as LA

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
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance

def mat_el_from_fids(fid1, fid2, theta_R):
    r = np.sqrt(fid1)
    theta = np.arccos(((4.0 * fid2) - fid1 - 1.0) / (2.0 * r)) + theta_R
    #theta = np.arccos(((4.0 * fid2) - fid1 - 1.0) / (2.0 * r))
    print("cos term: ",np.cos(theta), "sin term: ",np.sin(theta))
    mat_el = r * np.exp(1j*theta)
    return mat_el
    
def get_result(f1_circuit, f2_circuit, instance):
    # Measure probabilities
    f1_result = instance.execute(f1_circuit)
    print(f1_result)

    # Get the first component of the statevector for more accurate
    # probabilities
    #print("First component of the statevector: ", f1_result.get_statevector()._data[0])
    fid1 = f1_result.get_statevector().probabilities()[0]
    print("Probability from statevector: ", fid1)

    f2_result = instance.execute(f2_circuit)
    print(f2_result)

    # Get the first component of the statevector for more accurate
    # probabilities
    #print("First component of the statevector: ", f2_result.get_statevector()._data[0])
    #val = f2_result.get_statevector()._data[0]
    #print("Prob of first component: ", np.abs(val)**2)
    fid2 = f2_result.get_statevector().probabilities()[0]
    print("Probability from statevector: ", fid2)
    
    return (fid1, fid2)

xyz = '''H 0.0 0.0 0.0
         H 1.5 0.0 0.0
         H 3.0 0.0 0.0
         H 4.5 0.0 0.0
         H 6.0 0.0 0.0
         H 7.5 0.0 0.0'''

# First, perform an RHF calculation using the qiskit_nature PySCF driver
driver = PySCFDriver(atom=xyz, charge=0, spin=0, method=MethodType.RHF, basis='sto3g')
driver_result = driver.run()
electronic_en = driver_result.get_property(ElectronicEnergy)
nuc_rep_en = electronic_en.nuclear_repulsion_energy
print("Nuclear repulsion energy: ", nuc_rep_en)

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

# Creating <0|H|0> by adding coefficients of the identities
theta_0=0.0
for op in hamiltonian.to_pauli_op():
    p = op.primitive
    if p.x.any() or p.z.any():
        continue
    else:
        print("identity: ",p)
        theta_0 += op.coeff
print("Coeffs of the identities: ", theta_0)

# Set up the number of timesteps and step size
time_steps=3
tau=0.1
# Samples=1 should give statevector results from qasm simulator
samples=1

# Initialize F, S matrices
F_mat = np.zeros((time_steps, time_steps), dtype=complex)
S_mat = np.zeros((time_steps, time_steps), dtype=complex)

for m in range(time_steps):
    # Create a unitary by exponentiating the Hamiltonian
    # And trotterizing onto the circuit
    #evolution = PauliTrotterEvolution()
    evolution = MatrixEvolution()
    scaled_hamiltonian = m * tau * hamiltonian

    # Create a unitary by exponentiating the Hamiltonian
    unitary = evolution.convert(scaled_hamiltonian.exp_i())

    # Create a QuantumInstance
    instance = QuantumInstance(backend=Aer.get_backend('statevector_simulator'), shots=samples)

    # For F1, create circuit |HF> and add the unitary
    qr1 = QuantumRegister(n_so)
    cr1 = ClassicalRegister(n_so)
    f1_circuit = QuantumCircuit(qr1,cr1)
    f1_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)

    f1_circuit.compose(unitary.to_circuit(), inplace=True)

    # Add another HartreeFock instance to make measurement easy
    f1_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)

    # To obtain F2, prepare the unitary circuit that creates the superposition
    # 1/sqrt(2) (|0>_n + |HF>)
    qr2 = QuantumRegister(n_so)
    cr2 = ClassicalRegister(n_so)
    f2_circuit = QuantumCircuit(qr2,cr2) 

    f2_circuit.h(0)
    for n in range(n_alpha-1):
        f2_circuit.cnot(n, n+1)
    f2_circuit.cnot(n_alpha-1, n_alpha*2)
    for n in range(n_alpha*2, (n_alpha*2)+n_beta-1):
        f2_circuit.cnot(n, n+1)

    f2_circuit.compose(unitary.to_circuit(), inplace=True)

    # Apply the gates in reverse to revert back to the |0>_n state
    # To make measurement easy
    for n in reversed(range(n_alpha*2, (n_alpha*2)+n_beta-1)):
        f2_circuit.cnot(n, n+1)
    f2_circuit.cnot(n_alpha-1, n_alpha*2)
    for n in reversed(range(n_alpha-1)):
        f2_circuit.cnot(n, n+1)
    f2_circuit.h(0)
    print(f2_circuit.draw())

    # Fidelities for S
    (fid1, fid2) = get_result(f1_circuit, f2_circuit, instance)

    theta_R = -1.0 * m * tau * theta_0
    print("Theta_R: ", theta_R)
    print("Fids: ", fid1, fid2)
    Smat_el = mat_el_from_fids(fid1, fid2, theta_R)

    # Filling the S matrix
    for n in range(time_steps-m):
        print("S_{}_{} (m={}) = {}".format(n, n+m, m, Smat_el))
        S_mat[n,n+m] = Smat_el
        print("S_{}_{} (m={}) = {}".format(n+m, n, m, np.conj(Smat_el)))
        S_mat[n+m,n] = np.conj(Smat_el)

    '''
    # Circuits for F
    Fmat_el = 0.0
    for h_op in hamiltonian:
        # Unitary corresponding to the single Ham operator
        ham_unitary = evolution.convert(h_op.exp_i())

        qr3 = QuantumRegister(n_so)
        cr3 = ClassicalRegister(n_so)
        f1h_circuit = QuantumCircuit(qr3,cr3)
        f1h_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)
        f1h_circuit.compose(ham_unitary.to_circuit(), inplace=True)
        f1h_circuit.compose(unitary.to_circuit(), inplace=True)
        f1h_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)
        
        qr4 = QuantumRegister(n_so)
        cr4 = ClassicalRegister(n_so)
        f2h_circuit = QuantumCircuit(qr4,cr4) 
        f2h_circuit.h(0)
        for n in range(n_alpha+n_beta-1):
            f2h_circuit.cnot(n, n+1)
        f2h_circuit.compose(unitary.to_circuit(), inplace=True)
        # Apply the gates in reverse to revert back to the |0>_n state
        # To make measurement easy
        for n in reversed(range(n_alpha+n_beta-1)):
            f2h_circuit.cnot(n, n+1)
        f2h_circuit.h(0)

        instance = QuantumInstance(backend=Aer.get_backend('aer_simulator_statevector'), shots=samples)
        (fid1, fid2) = get_result(f1h_circuit, f2h_circuit, instance)
        theta_R = -1.0 * m * tau * theta_0
        Fmat_el += mat_el_from_fids(fid1, fid2, theta_R)

    # Filling the F matrix
    for n in range(m):
        print("F_{}_{} (m={}) = {}".format(n, m, m, Fmat_el))
        F_mat[n,m] = Fmat_el
        print("F_{}_{} (m={}) = {}".format(m, n, m, np.conj(Fmat_el)))
        F_mat[m,n] = np.conj(Fmat_el)
print(F_mat)
'''
print(S_mat)

exit()
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
qksd_en_list = -1.0j * np.log(eigvals)/ tau
print("QKSD energy: ", qksd_en_list)

#print("Error for timesteps={}: {}".format(time_steps, qksd_en-np_en))
