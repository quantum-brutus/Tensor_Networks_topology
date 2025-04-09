import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Paramètres des graphes
num_graphs = 8  # Nombre de graphes
num_qubits = 6  # Nombre de qubits par graphe
num_edges = 8   # Nombre d'arêtes par graphe

# Stockage des graphes et des MST maximaux
graphs = []
mst_max_graphs = []

# Fixer la seed pour la reproductibilité
np.random.seed(42)

for _ in range(num_graphs):
    G = nx.Graph()
    
    # Génération aléatoire des arêtes avec poids (nombre de portes à 2 qubits)
    edges = []
    for _ in range(num_edges):
        q1, q2 = np.random.choice(num_qubits, 2, replace=False)
        weight = np.random.randint(1, 10)  # Poids entre 1 et 10
        edges.append((q1, q2, weight))
    
    G.add_weighted_edges_from(edges)
    
    # Calcul du Maximum Spanning Tree (MST maximal)
    mst_max = nx.maximum_spanning_tree(G, weight="weight", algorithm="prim")
    
    graphs.append(G)
    mst_max_graphs.append(mst_max)

# Affichage des graphes
fig, axes = plt.subplots(4, 4, figsize=(16, 16))

for i in range(num_graphs):
    row, col = divmod(i, 2)

    # Position des nœuds
    pos = nx.spring_layout(graphs[i])

    # Graphe initial (à gauche)
    axes[row, col * 2].set_title(f"Graphe Initial {i+1}")
    nx.draw(graphs[i], pos, with_labels=True, node_color="lightblue", edge_color="gray", width=2, ax=axes[row, col * 2])
    labels = nx.get_edge_attributes(graphs[i], "weight")
    nx.draw_networkx_edge_labels(graphs[i], pos, edge_labels=labels, ax=axes[row, col * 2])

    # Maximum Spanning Tree (MST Maximal) (à droite)
    axes[row, col * 2 + 1].set_title(f"Max MST {i+1}")
    nx.draw(mst_max_graphs[i], pos, with_labels=True, node_color="lightgreen", edge_color="red", width=2.5, ax=axes[row, col * 2 + 1])
    labels_mst_max = nx.get_edge_attributes(mst_max_graphs[i], "weight")
    nx.draw_networkx_edge_labels(mst_max_graphs[i], pos, edge_labels=labels_mst_max, ax=axes[row, col * 2 + 1])

plt.tight_layout()
plt.show()
