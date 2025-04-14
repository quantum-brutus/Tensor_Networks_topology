import matplotlib.pyplot as plt
import numpy as np
from os import listdir, chdir

print(listdir())
# chdir('Tensor_Networks_topology')  # adapte ce chemin si besoin
print(listdir())

# Chargement
results = np.loadtxt('results.txt')

# Extraction
N, CHI, RES_tree, RES_mps, RES_mps_ordered = zip(*results)

# Reshape
chi_range = range(1, 12)  # adapte si besoin
N_gate_range = range(10, 200, 30)  # pareil

N = np.array(N).reshape(len(N_gate_range), len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range), len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range), len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range), len(chi_range))
RES_mps_ordered = np.array(RES_mps_ordered).reshape(len(N_gate_range), len(chi_range))

# Différences
RES_diff_tree_mps = RES_tree - RES_mps
RES_diff_mps_ordered = RES_mps_ordered - RES_mps

# -------------------------------------
#               PLOT
# -------------------------------------

fig = plt.figure(figsize=(15, 10))

# 1. Tree fidelity
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(CHI, N, RES_tree, alpha=0.7, cmap='viridis')
ax1.set_title('Tree Fidelity')
ax1.set_xlabel('Chi')
ax1.set_ylabel('N')
ax1.set_zlabel('Fidelity')

# 2. MPS fidelity
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(CHI, N, RES_mps, alpha=0.7, cmap='plasma')
ax2.set_title('MPS Fidelity')
ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_zlabel('Fidelity')

# 3. MPS Ordered fidelity
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot_surface(CHI, N, RES_mps_ordered, alpha=0.7, cmap='inferno')
ax3.set_title('MPS Ordered Fidelity')
ax3.set_xlabel('Chi')
ax3.set_ylabel('N')
ax3.set_zlabel('Fidelity')

# 4. Diff (Tree - MPS)
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot_surface(CHI, N, RES_diff_tree_mps, alpha=0.7, cmap='bwr')
ax4.set_title('Fidelity Diff: Tree - MPS')
ax4.set_xlabel('Chi')
ax4.set_ylabel('N')
ax4.set_zlabel('Δ Fidelity')

# 5. Diff (MPS ordered - MPS)
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot_surface(CHI, N, RES_diff_mps_ordered, alpha=0.7, cmap='bwr')
ax5.set_title('Fidelity Gain (Ordered MPS - MPS)')
ax5.set_xlabel('Chi')
ax5.set_ylabel('N')
ax5.set_zlabel('Δ Fidelity')

plt.tight_layout()
plt.show()
