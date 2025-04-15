import quimb as qu
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from circuit_generation import *
from tree_class import *
from best_mps_ordering import local_greedy_swap, greedy_weighted_linear_arrangement, plot_graph_comparisons

def test_tree_mps(circ,n_qb = 10,max_bond=None,return_memory=False, plot = False):
    mps = qtn.CircuitMPS(n_qb,max_bond=max_bond)
    mps_ordered = qtn.CircuitMPS(n_qb,max_bond=max_bond)

    G, mst, adj = plot_connectivity_and_max_mst(circ, return_adjacency_matrix=True)

    ## from the graph we derive the best permutation for the MPS 
    perm = greedy_weighted_linear_arrangement(G)
    initial_order = greedy_weighted_linear_arrangement(G)
    perm, final_cost = local_greedy_swap(G, initial_order)

    perm = list(range(n_qb))
    perm = list(reversed(range(n_qb)))


    print("Permutation is", perm)
    print("Final cost is", final_cost)

    if plot : 
        plot_graph_comparisons(G, initial_order, perm)


    ## reverse the permutation for a good mapping of the gates
    inverse_perm = {q: i for i, q in enumerate(perm)}


    for gate in circ.gates:
        qubits = gate.qubits  # par exemple (2,) ou (0, 1)
        print("old qubits is", qubits)
        print("old gate is", gate)
        remapped_qubits = tuple(inverse_perm[q] for q in gate.qubits)

        mps.apply_gate(gate)
        new_gate = gate.copy()
        new_gate.qubits = np.int64(remapped_qubits)
        print("new qubits is", remapped_qubits)
        print("new_gate is", new_gate)
        mps_ordered.apply_gate(new_gate)

    #mps.psi.draw(color=['PSI0', 'H', 'CX', 'RZ', 'RX', 'CZ'], show_tags=True)

    tree,trad_dic = tree_from_matrix(adj,0,max_bond=max_bond)
    tree = replicate_circ(circ,tree,trad_dic)


    reindex_dic = {'k'+str(k): 'out'+ str(v) for k, v in trad_dic.items()}
    # circ_net = circ.psi.reindex_(reindex_dic)
    # tree_net = tree.network.contract()

    # fnet = circ_net.H & tree_net
    # fnetMPS = circ.psi.H & mps.psi
    # fidelity_tree = np.abs(fnet.contract())
    # fidelity_mps = np.abs(fnetMPS.contract())

    circ_net_reindexed = circ.psi.reindex_(reindex_dic).contract().H
    circ_net = circ.psi.contract().H
    tree_net = tree.network.contract()

    # Inverse de l'inverse : reconstitution du perm original
    reindex_dic_ordered = {'out'+str(i): 'out'+str(inverse_perm[i]) for i in range(n_qb)}

    circ_net_ordered = circ.psi.reindex_(reindex_dic_ordered).contract().H

    mps_net = mps.psi.contract()
    mps_ordered_net = mps_ordered.psi.contract()

    fnet = circ_net_reindexed & tree_net
    fnetMPS = circ_net & mps_net
    fnetMPSordered = circ_net_ordered & mps_ordered_net
    fidelity_tree = np.abs(fnet.contract())
    fidelity_mps = np.abs(fnetMPS.contract())
    fidelity_mps_ordered = np.abs(fnetMPSordered.contract())

    if return_memory:
        mem_tree = np.sum(tree_net.contract().data.shape)
        mem_mps = np.sum(mps.psi.contract().data.shape)
        mem_mps_ordered = np.sum(mps_ordered.psi.contract().data.shape)

        return(fidelity_tree,fidelity_mps,fidelity_mps_ordered, mem_tree,mem_mps, mem_mps_ordered)
    
    return(fidelity_tree, fidelity_mps, fidelity_mps_ordered)

def replicate_circ(circ,tree,trad_dic):
    
    for gate in circ.gates:
        if gate.label == 'RX':
            param = gate.params[0]
            qubit = trad_dic[gate.qubits[0]]
            tree.apply_1qb_gate(i = qubit,gate_array=gate.build_array())

        if gate.label == 'CNOT':
            qubit_1 = trad_dic[gate.qubits[0]]
            qubit_2 = trad_dic[gate.qubits[1]]
            tree.apply_2qb_gate(i = qubit_1,j = qubit_2,gate_array=gate.build_array().reshape(2,2,2,2))
    return tree

tree_fidelity_values = []
mps_fidelity_values = []
mps_ordered_fidelity_values = []
diff_values = []
delta_mps = []

chi_range = range(1,12) 
N_gate_range = range(10,200,30)

nb_moy = 10

# for chi in chi_range:
#     print(chi)
#     tree_fidelity,mps_fidelity = test_tree_mps(n_qb=15,n_gates=50,max_bond=chi)
#     tree_fidelity_values.append(tree_fidelity)
#     mps_fidelity_values.append(mps_fidelity)
#     diff_values.append(tree_fidelity-mps_fidelity)

results = []
chi_tree_moy_dic = {}
chi_mps_moy_dic = {}
chi_mps_ordered_moy_dic = {}

mem_moy_dic = {}
  
N_qubits = 5

for N in tqdm(N_gate_range):
    print(N)
    for chi in chi_range:
        chi_mps_moy_dic[chi] = []
        chi_mps_ordered_moy_dic[chi] = []
        chi_tree_moy_dic[chi] = []
        mem_moy_dic[chi] = []

    for i in range(nb_moy):
        circ = generate_random_circuit(num_qubits=N_qubits,num_gates=N,plot=False)

        for chi in chi_range:
            tree_fidelity,mps_fidelity, mps_ordered_fidelity = test_tree_mps(circ = circ,n_qb=N_qubits,max_bond=chi,return_memory=False)
            chi_tree_moy_dic[chi].append(tree_fidelity)
            chi_mps_moy_dic[chi].append(mps_fidelity)
            chi_mps_ordered_moy_dic[chi].append(mps_ordered_fidelity)

    
    for chi in chi_range:
        results.append((N,chi,np.mean(chi_tree_moy_dic[chi]),np.mean(chi_mps_moy_dic[chi]), np.mean(chi_mps_ordered_moy_dic[chi])))

np.savetxt('results.txt',results)


N,CHI,RES_tree,RES_mps, RES_mps_ordered = zip(*results)
# RES = np.array(RES).reshape(len(N),len(CHI))
N = np.array(N).reshape(len(N_gate_range),len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range),len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range),len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))
RES_mps_ordered = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))

RES_diff = np.subtract(RES_tree,RES_mps)
RES_delta_mps = np.subtract(RES_mps_ordered,RES_mps)
fig = plt.figure()

###
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(CHI, N, RES_tree, alpha = 0.7,cmap='coolwarm')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax1.set_xlabel('Chi')
ax1.set_ylabel('N')
ax1.set_zlabel('fidelity tree')


###
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(CHI, N, RES_mps, alpha = 0.7,cmap='coolwarm')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_zlabel('fidelity mps')

###

ax2 = fig.add_subplot(133, projection='3d')
ax2.plot_surface(CHI, N, RES_diff, alpha = 0.7,cmap='coolwarm')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_zlabel('fidelity diff')

ax3 = fig.add_subplot(133, projection='3d')
ax2.plot_surface(CHI, N, RES_delta_mps, alpha = 0.7,cmap='coolwarm')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_zlabel('fidelity difference for MPS')