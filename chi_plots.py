import matplotlib.pyplot as plt
import random
import numpy as np
import quimb as qu
import quimb.tensor as qtn

result1 = []
result2 = []
N = 64

def gen_gates(N=64, x=0.1, depth=10, seed=42):
    """Generate a long range circuit that only slowly builds up entanglement.

    Parameters
    ----------
    N : int, optional
        The number of qubits.
    x : float, optional
        The average angle magnitude of U3 rotations.
    depth : int, optional
        The number of fully entangling gate layers.
    seed : int, optional
        A random seed.

    Yields
    ------
    qtn.Gate
    """

    rng = np.random.default_rng(seed)
    qubits = list(range(N))

    for _ in range(depth):
        # random small single qubit rotations
        for q in qubits:
            yield qtn.Gate("U3", params=rng.normal(scale=x, size=3), qubits=[q])

        # random CZs between arbitrary qubit pairs
        rng.shuffle(qubits)
        for i in range(0, N, 2):
            qa = qubits[i]
            qb = qubits[i + 1]
            yield qtn.Gate("CZ", params=(), qubits=[qa, qb])

for i in range(1,200,10):
    print('chi : ',i)
    chi = i
    circ = qtn.CircuitMPS(N,max_bond=chi)
    gates_generator = gen_gates(N=N)
    for gate in list(gates_generator):
        circ.apply_gate(gate)
    # circ.psi.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
    result1.append(circ.error_estimate())

for i in range(1,64,10):
    print('N : ',i )
    chi = 50
    N = i
    circ = qtn.CircuitMPS(N,max_bond=chi)
    gates_generator = gen_gates(N=N)
    for gate in list(gates_generator):
        circ.apply_gate(gate)
    # circ.psi.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'])
    result2.append(circ.error_estimate())



fig,axes = plt.subplots(2)

axs[0].plot(result1)
axs[1].plot(result2)

axs[0].set_yscale('log')
axs[1].set_yscale('log')

plt.show()

np.savetxt('result1.txt', result1, delimiter=',')
np.savetxt('result2.txt', result2, delimiter=',')