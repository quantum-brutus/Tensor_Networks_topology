import networkx as nx
import matplotlib.pyplot as plt

# Étape 1: Définir le graphe de connectivité des qubits
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

# 🔹 Étape 2: Construire l'arbre couvrant minimal (MST) avec l'algorithme de Kruskal
mst = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

# 🔹 Étape 3: Affichage des graphes
plt.figure(figsize=(10, 5))

# 🔹 Affichage du graphe original
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G)  # Disposition du graphe
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", width=2)
labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Graphe de connectivité initial")

# 🔹 Affichage du MST (arbre couvrant minimal)
plt.subplot(1, 2, 2)
nx.draw(mst, pos, with_labels=True, node_color="lightgreen", edge_color="red", width=2.5)
labels_mst = nx.get_edge_attributes(mst, "weight")
nx.draw_networkx_edge_labels(mst, pos, edge_labels=labels_mst)
plt.title("Arbre couvrant minimal (MST)")

plt.show()

