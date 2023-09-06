import numpy as np

def make_quantum_state(n):
    '''
    Create a quantum state of n qubits, initialized to the zero state.
    '''
    s = np.zeros(2**n)
    s[0] = 1
    return s

def qubit_dimension(d):
    '''
    Return the number of quibits represetend by the dimension d.
    '''
    return int(np.log2(d))

def apply_operator(matrix, column):
    '''
    Matrix-vector multiplication of a quantum operator with a quantum state.
    '''
    return np.dot(matrix, column)

def compose_operators(A, B):
    '''
    Compose two quantum operators A and B.
    '''
    return np.dot(A, B)

def sample(state):
    '''
    Sample a bit from a quantum state.
    '''
    r = np.random.rand()
    for i in range(len(state)):
        r -= np.abs(state[i])**2
        if r <= 0:
            return i

def collapse(state, bit):
    '''
    Collapse a quantum state to a specific bit.
    '''
    state[:] = 0.
    state[bit] = 1.
    return state


def observe(machine):
    b = sample(machine.quantum_state)
    machine.quantum_state =  collapse(machine.quantum_state, b)
    machine.quantum_state_register = b


def apply_gate(state, U, qubits):
    '''
    Apply a quantum gate to a quantum state.
    '''
    n = qubit_dimension(U.shape[0])
    if len(qubits) != n:
        raise ValueError('The number of qubits does not match the dimension of the quantum state.')
    if len(qubits) == 1:
        new_state = apply_1Q_gate(state, U, qubits[0])
    else:
        new_state = apply_nQ_gate(state, U, qubits)
    return new_state
    
def kronecker_product(A, B):
    '''
    Compute the Kronecker product of two matrices A and B.
    '''
    return np.kron(A, B)

def kronecker_exp(U, n):
    '''
    Compute the Kronecker product of a matrix U with itself n times.
    '''
    if n < 1:
        return np.ones((1, 1))
    if n == 1:
        return U
    else:
        return kronecker_product(U, kronecker_exp(U, n-1))

def lift(U, i, n):
    '''
    Lift a matrix U to the i-th qubit of a quantum state of n qubits.
    --> Remember we count quibits from the right side!!
    '''
    left = kronecker_exp(identity_gate, n-i-qubit_dimension(U.shape[0]))
    right = kronecker_exp(identity_gate, i)

    return kronecker_product(left, kronecker_product(U, right))


def apply_1Q_gate(state, U, qubit):
    '''
    Apply a quantum gate to a quantum state of 1 qubit.
    '''
    n = qubit_dimension(len(state))
    return apply_operator(lift(U, qubit, n), state)
    

def permutations_to_transpositions(permutation):
    '''
    Take a permutation list and convert it to a list of transpositions to be applied left-to-right.
    '''
    transpositions = []
    for dest in range(len(permutation)):
        src = permutation[dest]
        while src < dest:
            src = permutation[src]
        if src < dest:
            transpositions.append((src, dest))
        elif src > dest:
            transpositions.append((dest, src))
    return transpositions

def transpostions_to_adj_transpositions(transpositions):
    '''
    Take the transpositions pairs and generate a list of adjacent transpositions.
    '''
    adj_transpositions = []
    for el in transpositions:
        if el[0] < el[1]:
            adj_transpositions.extend([i for i in range(el[0], el[1])])
            adj_transpositions.extend([i for i in range(el[1]-2, el[0]-1, -1)])
    return adj_transpositions

def apply_nQ_gate(state, U, qubits):
    '''
    Apply a quantum gate to a quantum state of n qubits.
    '''
    n = qubit_dimension(len(state))
    def swap(i):
        return lift(SWAP, i, n)
    def transposition_to_operator(trans):
        if len(trans) > 1:
            tmp = compose_operators(swap(trans[0]), swap(trans[1]))
            for i in range(2,len(trans)):
                tmp = compose_operators(tmp, swap(trans[i]))
            return tmp
        elif len(trans) == 1:
            return swap(trans[0])
        else:
            return kronecker_exp(identity_gate, n)
    U01 = lift(U, 0, n)
    # from_space contains the permutation p encoding the space where we want to work
    from_space = [q for q in reversed(qubits)]
    for i in range(n):
        if i not in qubits:
            from_space.append(i)
    trans = transpostions_to_adj_transpositions(permutations_to_transpositions(from_space))
    to_from = transposition_to_operator(trans) # transposing to the space where we want to work
    from_to = transposition_to_operator(trans[::-1]) # going back to the original space
    Upq = compose_operators(to_from, compose_operators(U01, from_to))
    return apply_operator(Upq, state)


def run_quantum_program(qprog, machine):
    '''
    Run the quantum program `qprog` on the quantum machine `machine`.
    '''
    for el in qprog:
        instruction = el[0]
        payload = el[1]
        if instruction == 'GATE':
            machine.quantum_state = apply_gate(machine.quantum_state, payload['OPERATOR'], payload['QUBITS'])
        elif instruction == 'MEASURE':
            observe(machine)
    return machine
    

    
# Quantum gates
identity_gate = np.array([[1, 0], [0, 1]])
NOT_gate = np.array([[0, 1], [1, 0]]) # X_gate
SWAP = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
Hadamard_gate = 1/np.sqrt(2)*np.array([[1, 1], [1, -1]])
CNOT_gate = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1],[0, 0, 1, 0]])
Y_gate = np.array([[0, -1j], [1j, 0]])
Z_gate = np.array([[1, 0], [0, -1]])
Rx = lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
Ry = lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
Rz = lambda theta: np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

# To create a machine with 2 qubits we can just write:
machine = type('obj', (object,), {'quantum_state' : make_quantum_state(2), 'quantum_state_register' : 0})

def pauli_y(p):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [p]}))
    circuit.append(('GATE',{'OPERATOR': Y_gate, 'QUBITS': [p]}))
    return circuit

def pauli_z(p):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [p]}))
    circuit.append(('GATE',{'OPERATOR': Z_gate, 'QUBITS': [p]}))
    return circuit

def rotations(thetax, thetay, thetaz, p):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Rx(thetax), 'QUBITS': [p]}))
    circuit.append(('GATE',{'OPERATOR': Ry(thetay), 'QUBITS': [p]}))
    circuit.append(('GATE',{'OPERATOR': Rz(thetaz), 'QUBITS': [p]}))
    return circuit


def bell(p,q):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [p]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [p,q]}))
    return circuit


def ghz(n):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [0]}))
    for i in range(1,n):
        circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [0,i]}))
    return circuit

def entanglement_swap_4q(a,b,c,d):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [a]}))
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [c]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [a,b]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [c,d]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [b,c]}))
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [b]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [c,d]}))
    return circuit

def entanglement_swap_3q(a,b,c):
    circuit = []
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [a]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [a,b]}))
    circuit.append(('GATE',{'OPERATOR': CNOT_gate, 'QUBITS': [b,c]}))
    circuit.append(('GATE',{'OPERATOR': Hadamard_gate, 'QUBITS': [b]}))
    return circuit

    