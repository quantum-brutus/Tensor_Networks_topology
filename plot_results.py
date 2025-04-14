import matplotlib.pyplot as plt
import numpy as np
from os import listdir,chdir

print(listdir())
chdir('Tensor_Networks_topology')
print(listdir())

results = np.loadtxt('results_chi_1-20_N_10-100.txt')
chi_range = range(1,20)
N_gate_range = range(10,100,10)

N,CHI,RES_tree,RES_mps,MEM_tree,MEM_mps = zip(*results)
# RES = np.array(RES).reshape(len(N),len(CHI))
N = np.array(N).reshape(len(N_gate_range),len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range),len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range),len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))
MEM_tree = np.array(MEM_tree).reshape(len(N_gate_range),len(chi_range))
MEM_mps = np.array(MEM_mps).reshape(len(N_gate_range),len(chi_range))

RES_diff = np.subtract(RES_tree,RES_mps)
MEM_diff = np.subtract(MEM_tree,MEM_mps)

fig = plt.figure(figsize=(5,5))

# --------------- Tree fidelity
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot_surface(CHI, N, RES_tree, alpha = 0.7,cmap='bwr')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='bwr')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='bwr')

ax1.set_xlabel('Chi')
ax1.set_ylabel('N')
ax1.set_title('Tree Fidelity')


# --------------- MPS fidelity
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot_surface(CHI, N, RES_mps, alpha = 0.7,cmap='bwr')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='bwr')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='bwr')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_title('MPS fidelity')

# --------------- Fidelity diff

ax2 = fig.add_subplot(233, projection='3d')
ax2.plot_surface(CHI, N, RES_diff, alpha = 0.7,cmap='bwr')

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='bwr')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='bwr')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N')
ax2.set_title('Fidelity Difference (Tree-MPS)')

# --------------- Nb coef Tree

ax3 = fig.add_subplot(234, projection='3d')
ax3.plot_surface(CHI,N ,MEM_tree ,alpha = 0.7,cmap='bwr')

ax3.set_xlabel('Chi')
ax3.set_ylabel('N')
ax3.set_title('Nb of Coef Tree')

# ------------ Nb Coef MPS

ax4 = fig.add_subplot(235, projection='3d')
ax4.plot_surface(CHI,N ,MEM_mps ,alpha = 0.7,cmap='bwr')

ax4.set_xlabel('Chi')
ax4.set_ylabel('N')
ax4.set_title('Nb of Coef MPS')

# ------------ Nb of Coef Diff

ax5 = fig.add_subplot(236, projection='3d')
ax5.plot_surface(CHI,N ,MEM_diff ,alpha = 0.7,cmap='bwr')

ax5.set_xlabel('Chi')
ax5.set_ylabel('N')
ax5.set_title('Nb of Coeff Difference (Tree-MPS)')

# # ------------ fidelity/Nb coef Tree

# ax6 = fig.add_subplot(337, projection='3d')
# ax6.plot_surface(CHI,MEM_tree ,RES_tree ,alpha = 0.7,cmap='bwr')

# ax6.set_xlabel('Chi')
# ax6.set_ylabel('Nb of Coef')
# ax6.set_title('Fidelity by Nb of coef (Tree)')

# # ------------ fidelity/Nb coef MPS

# ax7 = fig.add_subplot(338, projection='3d')
# ax7.plot_surface(CHI,MEM_mps ,RES_mps ,alpha = 0.7,cmap='bwr')

# ax7.set_xlabel('Chi')
# ax7.set_ylabel('Nb of Coef')
# ax7.set_title('Fidelity by Nb of coef (MPS)')


# --------
# fig.subplots_adjust(wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.show()