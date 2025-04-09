import networkx as nx
import matplotlib.pyplot as plt

# 🔹 Étape 1: Définir le graphe de connectivité des qubits (pondéré par le nombre d'opérations à 2 qubits)
edges = [
    (0, 1, 3),  # Qubit 0 <-> Qubit 1 avec 3 portes à 2 qubits
    (0, 2, 2),
    (1, 2, 4),
    (1, 3, 1),
    (2, 3, 5),
    (2, 4, 2),
    (3, 4, 3),
]  # Liste des arêtes avec (qubit1, qubit2, poids)

G = nx.Graph()
G.add_weighted_edges_from(edges)

# 🔹 Étape 2: Construire l'arbre couvrant de poids maximal (Maximum Spanning Tree)
mst_max = nx.maximum_spanning_tree(G, weight="weight", algorithm="kruskal")

# 🔹 Étape 3: Affichage des graphes
plt.figure(figsize=(10, 5))

# 🔹 Affichage du graphe initial
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G)  # Disposition du graphe
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=2)
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Graphe de connectivité initial")

# 🔹 Affichage du Maximum Spanning Tree (MST maximal)
plt.subplot(1, 2, 2)
nx.draw(mst_max, pos, with_labels=True, node_color="lightgreen", edge_color="red", width=2.5)
labels_mst_max = nx.get_edge_attributes(mst_max, "weight")
nx.draw_networkx_edge_labels(mst_max, pos, edge_labels=labels_mst_max)
plt.title("Arbre couvrant de poids maximal (Max MST)")

plt.show()
