import quimb as qu
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from circuit_generation import *
from tree_class import *
from best_mps_ordering import local_greedy_swap, greedy_weighted_linear_arrangement, plot_graph_comparisons

def test_tree_mps(circ,n_qb = 10,max_bond=None,return_memory=False, plot = False):

    import time 

    start_time = time.time()

    G, mst, adj = plot_connectivity_and_max_mst(circ, return_adjacency_matrix=True)

    kruskal_time = time.time() - start_time

    print(f"Kruskal solver time: {kruskal_time:.4f} seconds")
    # permuted MPS

        # derive 'best' permutation from the graph


    start_time = time.time()

    perm = greedy_weighted_linear_arrangement(G)
    initial_order = greedy_weighted_linear_arrangement(G)
    perm, final_cost = local_greedy_swap(G, initial_order)

    perm = {i: q for i, q in enumerate(perm)}
        # reverse the permutation for a good mapping of the gates
    # inverse_perm = {'k'+str(i): 'k'+str(perm[i]) for i in perm.keys()}
    inverse_perm = {perm[q]:q for q in perm.keys()}
    solver_time = time.time() - start_time
    print(f"Heuristic local solver time: {solver_time:.4f} seconds")

    # print('--------')
    # print(n_qb)
    # print(perm)
    # print(inverse_perm)

    mps_ordered = qtn.CircuitMPS(n_qb,max_bond=max_bond)

    start_time = time.time()
    for gate in circ.gates:
        #qubits = gate.qubits  # par exemple (2,) ou (0, 1)
        remapped_qubits = tuple(inverse_perm[q] for q in gate.qubits)

        # mps.apply_gate(gate)
        new_gate = gate.copy()
        new_gate.qubits = np.int64(remapped_qubits)
        mps_ordered.apply_gate(new_gate)

    # reverse effect of permutation
    reindex_dic_ordered = {'k'+str(i): 'k'+str(perm[i]) for i in perm.keys()}
    mps_unperm = mps_ordered.psi.reindex(reindex_dic_ordered)
    
    apply_time_ordered = time.time() - start_time
    print(f"Apply gate time for ordered mps: {apply_time_ordered:.4f} seconds")

    if plot : 
        plot_graph_comparisons(G, initial_order, perm)

    # non reordered mps-------------------------------
    mps = qtn.CircuitMPS(n_qb,max_bond=max_bond)

    start_time = time.time()

    for gate in circ.gates:
        mps.apply_gate(gate)

    mps_net = mps.psi.contract()

    apply_time = time.time() - start_time

    print(f"Apply gate time: {apply_time:.4f} seconds")
    # Tree -------------------------------------------

    start_time = time.time()
    tree,trad_dic = tree_from_matrix(adj,0,max_bond=max_bond)


    tree = replicate_circ(circ,tree,trad_dic)

    reindex_dic = {'k'+str(k): 'out'+ str(v) for k, v in trad_dic.items()}
    circ_net_reindexed = circ.psi.reindex_(reindex_dic).contract().H
    tree_net = tree.network.contract()

    apply_time_tree = time.time() - start_time  
    print(f"Apply gate time for tree: {apply_time_tree:.4f} seconds")

    # Compute fidelity to original circuit
    circ_net = circ.psi.contract().H

    fnet = circ_net_reindexed & tree_net
    fnetMPS = circ_net & mps_net
    fnetMPSordered = circ_net & mps_unperm
    
    fidelity_tree = np.abs(fnet.contract())
    fidelity_mps = np.abs(fnetMPS.contract())
    fidelity_mps_ordered = np.abs(fnetMPSordered.contract())

    if return_memory:
        mem_tree = sum(t.size for t in tree.network)
        mem_mps = sum(t.size for t in mps.psi)
        mem_omps = sum(t.size for t in mps_ordered.psi)

        return(fidelity_tree,fidelity_mps,fidelity_mps_ordered, mem_tree,mem_mps, mem_omps, kruskal_time, solver_time, apply_time, apply_time_ordered, apply_time_tree)
    
    return(fidelity_tree, fidelity_mps, fidelity_mps_ordered, kruskal_time, solver_time, apply_time, apply_time_ordered, apply_time_tree)

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


# --------------------------------------------------------------------------------------------------

tree_fidelity_values = []
mps_fidelity_values = []
mps_ordered_fidelity_values = []
diff_values = []
delta_mps = []

# chi_range = range(1,40,3) 
# N_gate_range = range(10,200,30)
chi_range = range(1,40,3) 
N_gate_range = range(10,200,30)
nb_moy = 10

results = []
chi_tree_moy_dic = {}
chi_mps_moy_dic = {}
chi_omps_moy_dic = {}

mem_tree_moy_dic = {}
mem_mps_moy_dic = {}
mem_omps_moy_dic = {}

kruskal_times_dic = {}
heuristic_times_dic = {}
gate_times_mps_dic = {}
gate_times_mps_ordered_dic = {}
gate_times_tree_dic = {}


N_qubits = 3

for N in tqdm(N_gate_range):
    for chi in chi_range:
        chi_mps_moy_dic[chi] = []
        chi_omps_moy_dic[chi] = []
        chi_tree_moy_dic[chi] = []
        mem_tree_moy_dic[chi] = []
        mem_mps_moy_dic[chi] = []
        mem_omps_moy_dic[chi] = []
        kruskal_times_dic[chi] = []
        heuristic_times_dic[chi] = []
        gate_times_mps_dic[chi] = []
        gate_times_mps_ordered_dic[chi] = []
        gate_times_tree_dic[chi] = []


    for i in range(nb_moy):
        circ = generate_random_circuit(num_qubits=N_qubits,num_gates=N,plot=False)

        for chi in chi_range:
            tree_fidelity,mps_fidelity,omps_fidelity,mem_tree,mem_mps,mem_omps, kruskal_time, solver_time, mps_gate_time, mps_ordered_gate_time, tree_ordered_gate_time = test_tree_mps(circ = circ,n_qb=N_qubits,max_bond=chi,return_memory=True)
            chi_tree_moy_dic[chi].append(tree_fidelity)
            chi_mps_moy_dic[chi].append(mps_fidelity)
            chi_omps_moy_dic[chi].append(omps_fidelity)
            mem_tree_moy_dic[chi].append(mem_tree)
            mem_mps_moy_dic[chi].append(mem_mps)
            mem_omps_moy_dic[chi].append(mem_omps)
            kruskal_times_dic[chi].append(kruskal_time)
            heuristic_times_dic[chi].append(solver_time)
            gate_times_mps_dic[chi].append(mps_gate_time)
            gate_times_mps_ordered_dic[chi].append(mps_ordered_gate_time)
            gate_times_tree_dic[chi].append(tree_ordered_gate_time)
    
    for chi in chi_range:
        results.append((N,chi,np.mean(chi_tree_moy_dic[chi]),np.mean(chi_mps_moy_dic[chi]), np.mean(chi_omps_moy_dic[chi]),np.mean(mem_tree_moy_dic[chi]),np.mean(mem_mps_moy_dic[chi]),np.mean(mem_omps_moy_dic[chi])))
np.savetxt('results.txt',results)


N,CHI,RES_tree,RES_mps,RES_omps,MEM_tree,MEM_mps,MEM_omps = zip(*results)

chi_range = range(int(np.min(CHI)),int(np.max(CHI))+1,int(CHI[1]-CHI[0]))
N_gate_range = range(int(np.min(N)),int(np.max(N))+1,int(N[len(chi_range)]-N[0]))

# RES = np.array(RES).reshape(len(N),len(CHI))
N = np.array(N).reshape(len(N_gate_range),len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range),len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range),len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))
RES_omps = np.array(RES_omps).reshape(len(N_gate_range),len(chi_range))
MEM_tree = np.array(MEM_tree).reshape(len(N_gate_range),len(chi_range))
MEM_mps = np.array(MEM_mps).reshape(len(N_gate_range),len(chi_range))
MEM_omps = np.array(MEM_omps).reshape(len(N_gate_range),len(chi_range))

RES_diff = np.subtract(RES_tree,RES_omps)
MEM_diff = np.subtract(MEM_tree,MEM_omps)

fig = plt.figure(figsize=(5,5))

elev = 40
azim = 135
rstride = 2
cstride = 2

# ---------------

ax1 = fig.add_subplot(231, projection='3d')

ax1.view_init(elev=elev, azim=azim)
ax1.plot_surface(CHI, N, RES_tree, alpha = 0.7,cmap='coolwarm')
ax1.plot_wireframe(CHI, N, RES_tree, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax1.set_xlabel('Chi')
ax1.set_ylabel('N')
ax1.set_title('fidelity tree')


# ---------------
ax2 = fig.add_subplot(232, projection='3d')

ax2.view_init(elev=elev, azim=azim)
ax2.plot_surface(CHI, N, RES_omps, alpha = 0.7,cmap='coolwarm')
ax2.plot_wireframe(CHI, N, RES_omps, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_title('fidelity mps')

# ---------------

ax3 = fig.add_subplot(233, projection='3d')

ax3.view_init(elev=elev, azim=azim)
ax3.plot_surface(CHI, N, RES_diff, alpha = 0.7,cmap='coolwarm')
ax3.plot_wireframe(CHI, N, RES_diff, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax3.set_xlabel('Chi')
ax3.set_ylabel('N')
ax3.set_title('fidelity diff')

# ---------------

ax4 = fig.add_subplot(234, projection='3d')

ax4.view_init(elev=elev, azim=azim)
ax4.plot_surface(CHI,N ,MEM_tree ,alpha = 0.7,cmap='coolwarm')
ax4.plot_wireframe(CHI, N, MEM_tree, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

ax4.set_xlabel('Chi')
ax4.set_ylabel('N')
ax4.set_title('tree mem usage')

# ------------

ax5 = fig.add_subplot(235, projection='3d')

ax5.view_init(elev=elev, azim=azim)
ax5.plot_surface(CHI,N ,MEM_omps ,alpha = 0.7,cmap='coolwarm')
ax5.plot_wireframe(CHI, N, MEM_omps, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

ax5.set_xlabel('Chi')
ax5.set_ylabel('N')
ax5.set_title('mps mem usage')

# ------------

ax6 = fig.add_subplot(236, projection='3d')

ax6.view_init(elev=elev, azim=azim)
ax6.plot_surface(CHI,N ,MEM_diff ,alpha = 0.7,cmap='coolwarm')
ax6.plot_wireframe(CHI, N, MEM_diff, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)


ax6.set_xlabel('Chi')
ax6.set_ylabel('N')
ax6.set_title('mps mem usage')

# -------

plt.tight_layout()
plt.show()

## Les plots de temps de calcul de l'algorithme de Kruskal et de l'heuristique

fig, ax = plt.subplots(figsize=(8, 5))

avg_kruskal = [np.mean(kruskal_times_dic[chi]) for chi in chi_range]
avg_heuristic = [np.mean(heuristic_times_dic[chi]) for chi in chi_range]

ax.plot(chi_range, avg_kruskal, label='Kruskal Tree Solver', marker='o')
ax.plot(chi_range, avg_heuristic, label='Heuristic MPS Ordering', marker='x')

ax.set_xlabel('Chi')
ax.set_ylabel('Average Solver Time (s)')
ax.set_title('Solver Time Comparison')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

## Les plots de temps de calcul pour les circuits 

fig, ax = plt.subplots(figsize=(8, 5))

avg_mps_apply = [np.mean(gate_times_mps_dic[chi]) for chi in chi_range]
avg_omps_apply = [np.mean(gate_times_mps_ordered_dic[chi]) for chi in chi_range]
avg_tree_apply = [np.mean(gate_times_tree_dic[chi]) for chi in chi_range]

ax.plot(chi_range, avg_mps_apply, label='Standard MPS apply_gate', marker='o')
ax.plot(chi_range, avg_omps_apply, label='Ordered MPS apply_gate', marker='s')
ax.plot(chi_range, avg_tree_apply, label='Tree apply_gate', marker='^')

ax.set_xlabel('Chi')
ax.set_ylabel('Average Apply Gate Time (s)')
ax.set_title('Apply Gate Time Comparison')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
