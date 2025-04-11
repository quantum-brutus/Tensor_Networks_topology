## the goal here is to generate circuits using quimb
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


print('done')

num_qubits = 15
num_layers = 10
seed = 47

def generate_semi_random_entangled_circuits(num_qubits, num_layers, seed = None, plot = False):
    ## 1ere idée : faire des circuits quantiques aléatoires intriqués fortement
    rng = np.random.default_rng(seed=seed)
    angles = 2 * np.pi * rng.random((num_layers, num_qubits))
    print(angles)

    # besoin du nombre de couche, de la seed du random 
    circ = qtn.Circuit(N=num_qubits)

    for i in range(num_layers) : 
        for qubit in range(num_qubits):
        ##rotations aléatoires sur les qubits de manière non aléatoires
            circ.rx(theta=angles[i][qubit], i=qubit)
        
        ##cnot sur les adjacents maintenant
        for qubit in range(num_qubits) : 
            circ.cnot(qubit, (qubit+1)%num_qubits)
            if qubit == 0 : 
                circ.cnot(qubit, (qubit+1)%num_qubits)


    if plot : 
        circ.psi.draw(color=['PSI0', 'RX', 'CX'])

    print(circ)
    return circ


#circuit = generate_semi_random_entangled_circuits(num_qubits=num_qubits, num_layers=num_layers)

def plot_connectivity_and_max_mst(circ, return_adjacency_matrix=False, plot = True):

    G = nx.Graph()
    edge_weights = defaultdict(int)

    for gate in circ.gates:
        if len(gate.qubits) == 2:
            q1, q2 = sorted(gate.qubits)
            edge_weights[(q1, q2)] += 1

    for (q1, q2), weight in edge_weights.items():
        G.add_edge(q1, q2, weight=weight)

    mst = nx.maximum_spanning_tree(G, weight="weight", algorithm="kruskal")
    pos = nx.spring_layout(G, seed=42)

    if plot : 
            
        # --- Affichage des graphes ---
        plt.figure(figsize=(10, 5))

        # Graphe original
        plt.subplot(1, 2, 1)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "weight"))
        plt.title("Graphe de connectivité initial")

        # Arbre couvrant maximal
        plt.subplot(1, 2, 2)
        nx.draw(mst, pos, with_labels=True, node_color="lightgreen", edge_color="red", width=2.5)
        nx.draw_networkx_edge_labels(mst, pos, edge_labels=nx.get_edge_attributes(mst, "weight"))
        plt.title("Arbre couvrant maximal (Kruskal)")

        plt.tight_layout()
        plt.show()

    # --- Matrice d’adjacence (optionnelle) ---
    if return_adjacency_matrix:
        # Tri pour garantir l'ordre canonique des nœuds
        ordered_nodes = sorted(mst.nodes)
        adj_matrix = nx.to_numpy_array(mst, nodelist=ordered_nodes, weight='weight')
        return G, mst, adj_matrix

    return G, mst

# G, mst, adj = plot_connectivity_and_max_mst(circ, return_adjacency_matrix=True)


def generate_random_entangled_circuits(num_qubits, num_layers, seed1=None, seed2=None, plot=False):

    # --- Génération des angles RX ---
    rng_angles = np.random.default_rng(seed=seed1)
    angles = 2 * np.pi * rng_angles.random((num_layers, num_qubits))

    # --- Génération des couples CNOT (ctrl ≠ tgt) ---
    rng_cnots = np.random.default_rng(seed=seed2)
    total_needed = num_layers * num_qubits
    pairs = []

    while len(pairs) < total_needed:
        raw = rng_cnots.integers(0, num_qubits, size=(total_needed * 2, 2))
        # Supprimer les cas où ctrl == tgt
        filtered = [tuple(p) for p in raw if p[0] != p[1]]
        pairs.extend(filtered)

    # Réorganiser par couches
    cnot_pairs = [
        pairs[i * num_qubits:(i + 1) * num_qubits]
        for i in range(num_layers)
    ]

    # --- Construction du circuit ---
    circ = qtn.Circuit(N=num_qubits)

    for i in range(num_layers):
        for qubit in range(num_qubits):
            circ.rx(theta=angles[i][qubit], i=qubit)

        for ctrl, tgt in cnot_pairs[i]:
            circ.cnot(ctrl, tgt)

    # --- Affichage optionnel ---
    if plot:
        circ.psi.draw(color=['PSI0', 'RX', 'CX'])

    print(circ)
    return circ


circ = generate_random_entangled_circuits(
    num_qubits=8,
    num_layers=5,
    seed1=123,
    seed2=453,
    plot=True
)

G, mst, adj = plot_connectivity_and_max_mst(circ, return_adjacency_matrix=True)
print("Matrice d'adjacence pondérée :\n", adj)

# qc = qtn.Circuit(3)
# gates = [
#         ('H', 0),
#         ('H', 1),
#         ('CNOT', 1, 2),
#         ('CNOT', 0, 2),
#         ('H', 0),
#         ('H', 1),
#         ('H', 2),
#     ]
# qc.apply_gates(gates)
# qc.psi