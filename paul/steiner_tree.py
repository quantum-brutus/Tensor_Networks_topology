import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Création du graphe quantique (connectivité entre qubits)
G = nx.Graph()
edges = [
    (0, 1, 3), (0, 2, 2), (1, 2, 4), (1, 3, 1),
    (2, 3, 5), (2, 4, 2), (3, 4, 3), (3, 5, 2), (4, 5, 4)
]
G.add_weighted_edges_from(edges)

# Définition des qubits utilisés dans des opérations à 2 qubits
terminals = {0,1,2, 3, 4,5}

# Utilisation d'une approximation de Steiner Tree maximisant le poids total
steiner_tree = nx.maximum_spanning_tree(G.subgraph(terminals), weight="weight", algorithm="kruskal")

# Affichage des graphes
plt.figure(figsize=(10, 5))
pos = nx.spring_layout(G)

# Graphe complet
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=2)
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Graphe quantique initial")

# Arbre de Steiner maximisé
plt.subplot(1, 2, 2)
nx.draw(steiner_tree, pos, with_labels=True, node_color="lightgreen", edge_color="red", width=2.5)
labels_steiner = nx.get_edge_attributes(steiner_tree, "weight")
nx.draw_networkx_edge_labels(steiner_tree, pos, edge_labels=labels_steiner)
plt.title("Arbre de Steiner maximisé pour les qubits utilisés")

plt.show()
