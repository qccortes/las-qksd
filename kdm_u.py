# Script to use KDM/FDM Krylov subspace methods to
# compute ground state energies
# Currently only computes KDM U

import numpy as np
import scipy.linalg as LA

# Qiskit imports
from qiskit_nature.drivers.second_quantization import PySCFDriver, MethodType
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.circuit.library.initial_states import HartreeFock
from qiskit_nature.properties.second_quantization.electronic import ParticleNumber, ElectronicEnergy
from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import PauliTrotterEvolution
from qiskit.providers.aer import StatevectorSimulator, QasmSimulator
from qiskit import Aer, transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import QuantumInstance

xyz = '''H 0.0 0.0 0.0
         H 1.5 0.0 0.0
         H 3.0 0.0 0.0
         H 4.5 0.0 0.0
         H 6.0 0.0 0.0
         H 7.5 0.0 0.0'''

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

# Creating <0|H|0> by adding coefficients of the identities
theta_R=0.0
for op in hamiltonian.to_pauli_op():
    p = op.primitive
    if p.x.any() or p.z.any():
        continue
    else:
        print("identity: ",p)
        theta_R += op.coeff
print("Coeffs of the identities: ", theta_R)

# Set up the number of timesteps and step size
time_steps=6
tau=0.1
# Samples=1 should give statevector results from qasm simulator
samples=1

# Initialize F, S matrices
F_mat = np.zeros((time_steps, time_steps), dtype=complex)
S_mat = np.zeros((time_steps, time_steps), dtype=complex)
np.fill_diagonal(S_mat, 1.0)

for m in range(1,time_steps+1):
    # Create a unitary by exponentiating the Hamiltonian
    # And trotterizing onto the circuit
    evolution = PauliTrotterEvolution()
    scaled_hamiltonian = m * tau * hamiltonian

    unitary = evolution.convert(scaled_hamiltonian.exp_i())

    # For F1, create circuit |HF> and add the unitary
    qr1 = QuantumRegister(n_so)
    cr1 = ClassicalRegister(n_so)
    f1_circuit = QuantumCircuit(qr1,cr1)
    f1_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)
    print(f1_circuit.draw())

    f1_circuit.compose(unitary.to_circuit(), inplace=True)

    # Add another HartreeFock instance to make measurement easy
    f1_circuit.compose(HartreeFock(n_so, (n_alpha, n_beta), qubit_converter), inplace=True)

    # Measure probabilities
    f1_circuit.measure(range(n_so), range(n_so))

    #op_dict = circuit.decompose().decompose().count_ops()
    #total_ops = sum(op_dict.values())
    #print("Operations: {}".format(op_dict))
    #print("Total operations: {}".format(total_ops))
    #print("Nonlocal gates: {}".format(circuit.num_nonlocal_gates()))
    # Create a QuantumInstance
    instance = QuantumInstance(backend=Aer.get_backend('aer_simulator_statevector'), shots=samples)

    f1_result = instance.execute(f1_circuit)
    print(f1_result)

    # F1 is the probability of 0x0
    # (This probability includes shot noise)
    #fid1_meas = f1_result.get_counts()['000000000000'] / samples
    #print("Probability of 0x0: ", fid1_meas)

    # Get the first component of the statevector for more accurate
    # probabilities
    fid1 = f1_result.get_statevector().probabilities()[0]
    print("Probability from statevector: ", fid1)

    # To obtain F2, prepare the unitary circuit that creates the superposition
    # 1/sqrt(2) (|0>_n + |HF>)
    qr2 = QuantumRegister(n_so)
    cr2 = ClassicalRegister(n_so)
    f2_circuit = QuantumCircuit(qr2,cr2) 

    f2_circuit.h(0)
    for n in range(n_alpha+n_beta-1):
        f2_circuit.cnot(n, n+1)
    print(f2_circuit.draw())

    f2_circuit.compose(unitary.to_circuit(), inplace=True)

    # Apply the gates in reverse to revert back to the |0>_n state
    # To make measurement easy
    for n in reversed(range(n_alpha+n_beta-1)):
        f2_circuit.cnot(n, n+1)
    f2_circuit.h(0)

    # Measure probabilities
    f2_circuit.measure(range(n_so), range(n_so))

    f2_result = instance.execute(f2_circuit)
    print(f2_result)
    # F2 is the probability of 0x0 divided by 4
    # (This probability includes shot noise)
    #fid2_meas = 0.25 * f2_result.get_counts()['000000000000'] / samples
    #print("Probability of 0x0: ", fid2_meas)

    # Get the first component of the statevector for more accurate
    # probabilities
    fid2 = 0.25 * f2_result.get_statevector().probabilities()[0]
    print("Probability from statevector: ", fid2)

    r = np.sqrt(fid1)
    theta_R = -1.0 * m * tau * theta_R
    theta = np.arccos((4.0 * fid2 - fid1 - 1.0) / (2.0 * np.sqrt(fid1))) + theta_R

    mat_el = r * np.exp(1j*theta)
    print("Final matrix element for m={}: ".format(m), mat_el)

    # Filling the F matrix
    if m == 1:
        np.fill_diagonal(F_mat, mat_el)
    else:
        for n in range(time_steps-m+1):
            print("F_{}_{} (m={}) = {}".format(n, n+m-1, m, mat_el))
            F_mat[n,n+m-1] = mat_el
            print("F_{}_{} (m={}) = {}".format(n+m-1, n, m, mat_el))
            F_mat[n+m-1,n] = mat_el
    
    # Filling the S matrix
    for n in range(time_steps-m):
        print("S_{}_{} (m={}) = {}".format(n, n+m, m, mat_el))
        S_mat[n,n+m] = mat_el
        print("S_{}_{} (m={}) = {}".format(n+m, n, m, mat_el))
        S_mat[n+m,n] = mat_el

print(F_mat)
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
qksd_en_list = -1.0j * np.log(eigvals)/ tau
print("QKSD energy: ", qksd_en_list)

#print("Error for timesteps={}: {}".format(time_steps, qksd_en-np_en))
