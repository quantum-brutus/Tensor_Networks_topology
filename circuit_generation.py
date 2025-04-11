import quimb as qu
import quimb.tensor as qtn
import numpy as np
import networkx as nx
from collections import defaultdict


def generate_semi_random_entangled_circuits(num_qubits, num_layers, seed = None, plot = False):
    ## 1ere idée : faire des circuits quantiques aléatoires intriqués fortement
    rng = np.random.default_rng(seed=seed)
    angles = 2 * np.pi * rng.random((num_layers, num_qubits))
    # print(angles)

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

    # print(circ)
    return circ


#circuit = generate_semi_random_entangled_circuits(num_qubits=num_qubits, num_layers=num_layers)

def plot_connectivity_and_max_mst(circ, return_adjacency_matrix=False, plot = False):

    G = nx.Graph()
    edge_weights = defaultdict(int)

    # make sur all qubits are part of the graph/tree
    for q1 in range(circ.N):
        for q2 in range(circ.N):
            edge_weights[(q1, q2)] += 1

    for gate in circ.gates:
        if len(gate.qubits) == 2:
            q1, q2 = sorted(gate.qubits)
            edge_weights[(q1, q2)] += 5

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
        ordered_nodes = sorted(G.nodes)
        adj_matrix = nx.to_numpy_array(mst, nodelist=ordered_nodes, weight='weight')
        return G, mst, adj_matrix

    return G, mst

#G, mst, adj = plot_connectivity_and_max_mst(circuit, return_adjacency_matrix=True)
#print("Matrice d'adjacence pondérée :\n", adj)


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

    # print(circ)
    return circ

def generate_random_circuit(num_qubits,num_gates,seed1=None,seed2=None,plot=False):
    circ = qtn.Circuit(N=num_qubits)

    rng_angles = np.random.default_rng(seed=seed1)
    angles = rng_angles.random(num_gates)

    rng_cnots = np.random.default_rng(seed=None)
    cnots = []
    
    while len(cnots) < num_gates:
        ctrl,tgt = rng_cnots.integers(low=0,high=num_qubits,size = 2)
        if ctrl != tgt:
            cnots.append([ctrl,tgt])

    for i in range(num_gates):
        ctrl = cnots[i][0]
        tgt = cnots[i][1]
        # print(ctrl,tgt)

        circ.rx(theta=angles[i], i=tgt)
        circ.cnot(ctrl, tgt)
    
    if plot:
        circ.psi.draw(color=['PSI0', 'RX', 'CX'],iterations=100, k=6)
    return circ
