import matplotlib.pyplot as plt
import numpy as np
from os import listdir,chdir
import pandas as pd

print(listdir())
chdir('Tensor_Networks_topology\\mps_ordering')
print(listdir())

# results = np.loadtxt('results_df_10qb_30moy_chi_5-45-5_N_30-300-30.csv')
df_results = pd.read_csv('results_df.csv') 

N = df_results['N']
CHI = df_results['chi']
RES_tree = df_results['f_tree']
RES_mps = df_results['f_mps']
# RES_omps = df_results['f_omps']
MEM_tree = df_results['mem_tree']
MEM_mps = df_results['mem_mps']
# MEM_omps = df_results['mem_omps']

chi_range = range(int(np.min(CHI)),int(np.max(CHI))+1,int(CHI[1]-CHI[0]))
N_gate_range = range(int(np.min(N)),int(np.max(N))+1,int(N[len(chi_range)]-N[0]))

# RES = np.array(RES).reshape(len(N),len(CHI))
N = np.array(N).reshape(len(N_gate_range),len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range),len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range),len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))
# RES_omps = np.array(RES_omps).reshape(len(N_gate_range),len(chi_range))
MEM_tree = np.array(MEM_tree).reshape(len(N_gate_range),len(chi_range))
MEM_mps = np.array(MEM_mps).reshape(len(N_gate_range),len(chi_range))
# MEM_omps = np.array(MEM_omps).reshape(len(N_gate_range),len(chi_range))

RES_diff = np.subtract(RES_tree,RES_mps)
MEM_diff = np.subtract(MEM_tree,MEM_mps)
# RES_diff = np.subtract(RES_mps,RES_omps)
# MEM_diff = np.subtract(MEM_mps,MEM_omps)

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
ax1.set_ylabel('N_gates')
ax1.set_title('Fidelity (Tree)')


# ---------------
ax2 = fig.add_subplot(232, projection='3d')

ax2.view_init(elev=elev, azim=azim)
ax2.plot_surface(CHI, N, RES_mps, alpha = 0.7,cmap='coolwarm')
ax2.plot_wireframe(CHI, N, RES_mps, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax2.set_xlabel('Chi')
ax2.set_ylabel('N_gates')
ax2.set_title('Fidelity (MPS)')

# ---------------

ax3 = fig.add_subplot(233, projection='3d')

ax3.view_init(elev=elev, azim=azim)
ax3.plot_surface(CHI, N, RES_diff, alpha = 0.7,cmap='coolwarm')
ax3.plot_wireframe(CHI, N, RES_diff, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

# ax.contour(CHI, N, RES, zdir='x', offset=-15, cmap='coolwarm')
# ax.contour(CHI, N, RES, zdir='y', offset=500, cmap='coolwarm')

ax3.set_xlabel('Chi')
ax3.set_ylabel('N_gates')
ax3.set_title('Fidelity difference (Tree-MPS)')

# ---------------

ax4 = fig.add_subplot(234, projection='3d')

ax4.view_init(elev=elev, azim=azim)
ax4.plot_surface(CHI,N ,MEM_tree ,alpha = 0.7,cmap='coolwarm')
ax4.plot_wireframe(CHI, N, MEM_tree, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

ax4.set_xlabel('Chi')
ax4.set_ylabel('N_gates')
ax4.set_title('Nb. of Complex coef. (Tree)')

# ------------

ax5 = fig.add_subplot(235, projection='3d')

ax5.view_init(elev=elev, azim=azim)
ax5.plot_surface(CHI,N ,MEM_mps ,alpha = 0.7,cmap='coolwarm')
ax5.plot_wireframe(CHI, N, MEM_mps, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)

ax5.set_xlabel('Chi')
ax5.set_ylabel('N')
ax5.set_title('Nb. of Complex coef. (MPS)')

# ------------

ax6 = fig.add_subplot(236, projection='3d')

ax6.view_init(elev=elev, azim=azim)
ax6.plot_surface(CHI,N ,MEM_diff ,alpha = 0.7,cmap='coolwarm')
ax6.plot_wireframe(CHI, N, MEM_diff, rstride=rstride, cstride=cstride, color='black', linewidth=0.3)


ax6.set_xlabel('Chi')
ax6.set_ylabel('N')
ax6.set_title('Nb. of Complex coef. (Tree-MPS)')

# -------

fig.suptitle('Tree and MPS comparison')

plt.tight_layout()
plt.show()