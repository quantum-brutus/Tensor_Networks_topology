##best algorithm for the problem found (best tradoff time/quality)
import time
import networkx as nx
import random
import numpy as np

def local_greedy_swap(G, ordering, max_iter=100):
    """
    Perform a local greedy swap optimization on the ordering of nodes in the graph G. Takes in arguments:
    G: the graph to optimize
    ordering: the initial ordering of nodes
    max_iter: the maximum number of iterations to perform

    Returns the best ordering found and its corresponding cost.
    """
    best_order = ordering[:]
    best_cost = compute_total_cost(G, best_order)

    for _ in range(max_iter):
        improved = False
        for i in range(len(best_order)):
            for j in range(i + 1, len(best_order)):
                new_order = best_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_cost = compute_total_cost(G, new_order)
                if new_cost < best_cost:
                    best_order, best_cost = new_order, new_cost
                    improved = True
        if not improved:
            break
    return best_order, best_cost

def compute_total_cost(G, ordering):
    pos = {node: i for i, node in enumerate(ordering)}
    return sum(G[u][v]['weight'] * abs(pos[u] - pos[v]) for u, v in G.edges())

def greedy_weighted_linear_arrangement(G):
    nodes = list(G.nodes())
    placed = []
    remaining = set(nodes)

    # On commence par le nœud de plus haut degré pondéré
    start = max(remaining, key=lambda n: sum(G[n][nbr]['weight'] for nbr in G[n]))
    placed.append(start)
    remaining.remove(start)

    while remaining:
        # Pour chaque nœud restant, on évalue le gain à être proche des déjà placés
        best_node = None
        best_score = float('-inf')

        for node in remaining:
            score = sum(G[node][p]['weight'] for p in G[node] if p in placed)
            if score > best_score:
                best_node = node
                best_score = score

        placed.append(best_node)
        remaining.remove(best_node)

    return placed



import matplotlib.pyplot as plt

def plot_ordered_layout(G, ordering, title, ax):
    """
    Affiche un graphe linéaire selon un ordre donné,
    avec les arêtes et poids uniquement si elles existent dans G.
    """
    pos = {node: (i, 0) for i, node in enumerate(ordering)}
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=600)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    edge_labels = {}
    edges_to_draw = []

    for i in range(len(ordering) - 1):
        u, v = ordering[i], ordering[i + 1]
        if G.has_edge(u, v):
            weight = G[u][v]['weight']
            edges_to_draw.append((u, v))
            edge_labels[(u, v)] = weight

    # Tracer les arêtes valides
    nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw, ax=ax, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6)

    ax.set_title(title)
    ax.axis('off')


def plot_graph_comparisons(G, initial_order, final_order):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Graphe original
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos, ax=axs[0, 0],
        with_labels=True,
        node_color='lightgreen',
        edge_color='gray',
        node_size=600,
        font_size=8
    )
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=axs[0, 0], font_size=6)
    axs[0, 0].set_title("Graphe original (connectivité)")

    # 2. Greedy
    plot_ordered_layout(G, initial_order, f"Ordre greedy\nCoût: {compute_total_cost(G, initial_order)}", axs[0, 1])

    # 3. Local Swap
    plot_ordered_layout(G, final_order, f"Ordre local optimisé\nCoût: {compute_total_cost(G, final_order)}", axs[1, 0])

    plt.tight_layout()
    plt.show()

# # Création du graphe 15 qubits
# random.seed(400)
# G = nx.Graph()
# qubits = [i for i in range(15)]
# G.add_nodes_from(qubits)

# for _ in range(20):
#     u, v = random.sample(qubits, 2)
#     if G.has_edge(u, v):
#         continue
#     G.add_edge(u, v, weight=random.randint(1, 5))


# initial_order = greedy_weighted_linear_arrangement(G)

# # 2. Optimisation locale
# start = time.time()
# final_order, final_cost = local_greedy_swap(G, initial_order)
# local_time = time.time() - start

# print("Temps d'exécution de l'optimisation locale :", local_time)
# print("Ordre optimisé :", final_order)
# print("Coût optimisé :", final_cost)
