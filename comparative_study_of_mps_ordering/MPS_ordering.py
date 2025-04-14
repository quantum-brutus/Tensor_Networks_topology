import networkx as nx
import pulp

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

def compute_total_cost(G, ordering):
    pos = {node: i for i, node in enumerate(ordering)}
    return sum(G[u][v]['weight'] * abs(pos[u] - pos[v]) for u, v in G.edges())

def local_greedy_swap(G, ordering, max_iter=100):
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


def solve_mwla_ilp(G, timeout=10):
    import time
    nodes = list(G.nodes)
    n = len(nodes)
    positions = list(range(n))

    prob = pulp.LpProblem("MWLA", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (nodes, positions), cat='Binary')
    pi = pulp.LpVariable.dicts("pi", nodes, lowBound=0, upBound=n-1, cat='Integer')
    d = {(u, v): pulp.LpVariable(f"d_{u}_{v}", lowBound=0, cat='Integer') for u, v in G.edges}

    prob += pulp.lpSum(G[u][v]['weight'] * d[(u, v)] for u, v in G.edges)

    for u in nodes:
        prob += pulp.lpSum(x[u][p] for p in positions) == 1
    for p in positions:
        prob += pulp.lpSum(x[u][p] for u in nodes) == 1
    for u in nodes:
        prob += pi[u] == pulp.lpSum(p * x[u][p] for p in positions)
    for u, v in G.edges:
        prob += d[(u, v)] >= pi[u] - pi[v]
        prob += d[(u, v)] >= pi[v] - pi[u]

    # Solve with timeout
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
    prob.solve(solver)
    elapsed = time.time() - start

    ordering = sorted(nodes, key=lambda u: pulp.value(pi[u]))
    total_cost = pulp.value(prob.objective)

    return ordering, total_cost, elapsed




# G = nx.Graph()
# G.add_edge('q0', 'q1', weight=3)
# G.add_edge('q0', 'q2', weight=1)
# G.add_edge('q1', 'q2', weight=2)
# G.add_edge('q2', 'q3', weight=5)
# G.add_edge('q1', 'q3', weight=2)

import networkx as nx
import random


# G = nx.Graph()

# edges = [
#     ('q0', 'q1', 4),
#     ('q0', 'q2', 2),
#     ('q1', 'q3', 6),
#     ('q1', 'q4', 3),
#     ('q2', 'q4', 5),
#     ('q2', 'q5', 1),
#     ('q3', 'q6', 2),
#     ('q4', 'q6', 4),
#     ('q5', 'q6', 3),
#     ('q5', 'q7', 2),
#     ('q6', 'q7', 1)
# ]

# for u, v, w in edges:
#     G.add_edge(u, v, weight=w)


# Création du graphe 15 qubits
random.seed(402)
G = nx.Graph()
qubits = [i for i in range(15)]
G.add_nodes_from(qubits)

for _ in range(20):
    u, v = random.sample(qubits, 2)
    if G.has_edge(u, v):
        continue
    G.add_edge(u, v, weight=random.randint(1, 5))


# 1. Ordre greedy

import time

# 1. Greedy
start = time.time()
initial_order = greedy_weighted_linear_arrangement(G)
greedy_time = time.time() - start
print("Temps d'exécution de l'algorithme greedy :", greedy_time)
initial_cost = compute_total_cost(G, initial_order)


print("Ordre greedy :", initial_order)
print("Coût initial :", initial_cost)

# 2. Optimisation locale
# 2. Local Swap
start = time.time()
final_order, final_cost = local_greedy_swap(G, initial_order)
local_time = time.time() - start

print("Temps d'exécution de l'optimisation locale :", local_time)
print("Ordre optimisé :", final_order)
print("Coût optimisé :", final_cost)

# 3. ILP
ilp_order, cost, ilp_time = solve_mwla_ilp(G, timeout=1)
print("Ordre optimal ILP :", ilp_order)
print("Coût total :", cost)



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


def plot_graph_comparisons(G, initial_order, final_order, ilp_ordering):
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

    # 4. ILP
    plot_ordered_layout(G, ilp_ordering, f"Ordre optimal ILP\nCoût: {compute_total_cost(G, ilp_ordering)}", axs[1, 1])

    plt.tight_layout()
    plt.show()

plot_graph_comparisons(G, initial_order, final_order, ilp_order)

def plot_solver_times(greedy_time, local_time, ilp_time):
    import matplotlib.pyplot as plt

    labels = ['Greedy', 'Local Swap', 'ILP']
    times = [greedy_time, local_time, ilp_time]
    colors = ['skyblue', 'orange', 'lightcoral']

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, times, color=colors)
    plt.ylabel("Temps (secondes)")
    plt.title("Temps de calcul des solveurs")

    # Ajouter les durées au-dessus des barres
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width() / 2, t * 1.01, f"{t:.4f}s", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

plot_solver_times(greedy_time, local_time, ilp_time)
