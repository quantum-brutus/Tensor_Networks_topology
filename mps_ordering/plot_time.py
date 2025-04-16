import matplotlib.pyplot as plt
import numpy as np

# Chargement des données
results = np.loadtxt('results/results7qubits.txt')

# Extraction des colonnes
N, CHI, RES_tree, RES_mps, RES_omps, MEM_tree, MEM_mps, MEM_omps, \
time_kruskal, time_heuristic, time_gate_mps, time_gate_omps, time_gate_tree = zip(*results)

# Définition des plages
chi_range = range(1, 40, 3)
N_gate_range = range(10, 200, 30)

# Reshape
shape = (len(N_gate_range), len(chi_range))

N = np.array(N).reshape(shape)
CHI = np.array(CHI).reshape(shape)

# Temps reshape
T_kruskal = np.array(time_kruskal).reshape(shape)
T_heuristic = np.array(time_heuristic).reshape(shape)
T_gate_mps = np.array(time_gate_mps).reshape(shape)
T_gate_omps = np.array(time_gate_omps).reshape(shape)
T_gate_tree = np.array(time_gate_tree).reshape(shape)

T_kruskal_ms = T_kruskal * 1e3
T_heuristic_ms = T_heuristic * 1e3


# 1. Calcul des bornes communes
z_min = min(T_gate_mps.min(), T_gate_omps.min(), T_gate_tree.min())
z_max = max(T_gate_mps.max(), T_gate_omps.max(), T_gate_tree.max())

z_min_solver = min(T_kruskal_ms.min(), T_heuristic_ms.min())
z_max_solver = max(T_kruskal_ms.max(), T_heuristic_ms.max())

print(z_max_solver, z_min_solver)

# Différences
diff_tree_vs_mps = T_gate_tree - T_gate_mps
diff_omps_vs_mps = T_gate_omps - T_gate_mps

# ---------------------------
#       PLOTTING
# ---------------------------
fig = plt.figure(figsize=(18, 12))

# 1. Temps Kruskal (Tree total) en ms
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(CHI, N, T_kruskal_ms, cmap='viridis', alpha=0.8)
ax1.text2D(0.5, 0.95, 'Tree Solver Total Time (Kruskal)', transform=ax1.transAxes, ha='center', fontsize=12)
ax1.set_xlabel('Chi')
ax1.set_ylabel('N Gates')
ax1.set_zlabel('Time (ms)')
ax1.set_zlim(z_min_solver, z_max_solver)

# 2. Temps Heuristic (Ordered MPS total) en ms
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(CHI, N, T_heuristic_ms, cmap='plasma', alpha=0.8)
ax2.text2D(0.5, 0.95, 'Ordered MPS Total Time (Heuristic solver)', transform=ax2.transAxes, ha='center', fontsize=12)
ax2.set_xlabel('Chi')
ax2.set_ylabel('N Gates')
ax2.set_zlabel('Time (ms)')
ax2.set_zlim(z_min_solver, z_max_solver)
# 3. Temps gate-by-gate MPS
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(CHI, N, T_gate_mps, cmap='Blues', alpha=0.8)
ax3.text2D(0.5, 0.95, 'MPS circuit calculation Time', transform=ax3.transAxes, ha='center', fontsize=12)
ax3.set_xlabel('Chi')
ax3.set_ylabel('N Gates')
ax3.set_zlabel('Time (s)')
ax3.set_zlim(z_min, z_max)

# 4. Temps gate-by-gate Ordered MPS
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(CHI, N, T_gate_omps, cmap='Oranges', alpha=0.8)
ax4.text2D(0.5, 0.95, 'Ordered MPS circuit calculation Time', transform=ax4.transAxes, ha='center', fontsize=12)
ax4.set_xlabel('Chi')
ax4.set_ylabel('N Gates')
ax4.set_zlabel('Time (s)')
ax4.set_zlim(z_min, z_max)

# 5. Temps gate-by-gate Tree
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(CHI, N, T_gate_tree, cmap='Greens', alpha=0.8)
ax5.text2D(0.5, 0.95, 'Tree circuit calculation Time', transform=ax5.transAxes, ha='center', fontsize=12)
ax5.set_xlabel('Chi')
ax5.set_ylabel('N Gates')
ax5.set_zlabel('Time (s)')
ax5.set_zlim(z_min, z_max)

# 6. Différences Tree - MPS
ax6 = fig.add_subplot(236, projection='3d')
ax6.plot_surface(CHI, N, diff_tree_vs_mps, cmap='bwr', alpha=0.8)
ax6.text2D(0.5, 0.95, 'Time Difference: Tree - MPS', transform=ax6.transAxes, ha='center', fontsize=12)
ax6.set_xlabel('Chi')
ax6.set_ylabel('N Gates')
ax6.set_zlabel('Δ Time (s)')
ax6.set_zlim(z_min, z_max)

plt.tight_layout()
plt.show()


# Trouve l'indice du plus grand chi
idx_max_chi = len(chi_range) - 1  # dernier élément

# Extraire les courbes pour le chi maximal
N_vals = N[:, idx_max_chi]  # N varie avec le nombre de portes
T_mps_maxchi = T_gate_mps[:, idx_max_chi]
T_omps_maxchi = T_gate_omps[:, idx_max_chi]
T_tree_maxchi = T_gate_tree[:, idx_max_chi]

# Tracé 2D
plt.figure(figsize=(10, 6))
plt.plot(N_vals, T_mps_maxchi, label='MPS', marker='o')
plt.plot(N_vals, T_omps_maxchi, label='Ordered MPS', marker='s')
plt.plot(N_vals, T_tree_maxchi, label='Tree', marker='^')
plt.xlabel('Nombre de portes (N)')
plt.ylabel('Temps (s)')
plt.title(f'Temps pour appliquer le circuit pour chi = {chi_range[idx_max_chi]}, et 7 qubits')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

