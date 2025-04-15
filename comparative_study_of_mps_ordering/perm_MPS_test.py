import random
import quimb as qu
import quimb.tensor as qtn

# 10 qubits and tag the initial wavefunction tensors
circ = qtn.Circuit(N=5)

permutation = [0, 1, 2, 3, 4]


# 8 rounds of entangling gates
for r in range(1, 5):

    # X-rotations
    for i in range(5):
        circ.apply_gate('RX', 1.234, permutation[i], gate_round=r)

        # odd pairs
    for i in range(1, 5, 2):
        circ.apply_gate('CZ', permutation[i], permutation[i + 1], gate_round=r)


circ.psi.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], show_tags=True)