import matplotlib.pyplot as plt
import numpy as np
from os import listdir,chdir

print(listdir())
chdir('Tensor_Networks_topology')
print(listdir())

results = np.loadtxt('results_10-34_N_70-150.txt')

N,CHI,RES_tree,RES_mps,MEM_tree,MEM_mps = zip(*results)

chi_range = range(int(np.min(CHI)),int(np.max(CHI))+1,int(CHI[1]-CHI[0]))
N_gate_range = range(int(np.min(N)),int(np.max(N))+1,int(N[len(chi_range)]-N[0]))

# RES = np.array(RES).reshape(len(N),len(CHI))
N = np.array(N).reshape(len(N_gate_range),len(chi_range))
CHI = np.array(CHI).reshape(len(N_gate_range),len(chi_range))
RES_tree = np.array(RES_tree).reshape(len(N_gate_range),len(chi_range))
RES_mps = np.array(RES_mps).reshape(len(N_gate_range),len(chi_range))
MEM_tree = np.array(MEM_tree).reshape(len(N_gate_range),len(chi_range))
MEM_mps = np.array(MEM_mps).reshape(len(N_gate_range),len(chi_range))

i = 8
# plt.plot(MEM_tree[i],RES_tree[i],color='g')
# plt.plot(MEM_mps[i],RES_mps[i],color='r')
# plt.show()


fig = plt.figure()
gs = fig.add_gridspec(1, len(N.T[0]), hspace=0, wspace=0)
axes = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Fidelity by Nb of Coef for different number of Gates')
fig.supxlabel('Nb Coef')
fig.supylabel('Fidelity')

# fig, axes = plt.subplots(1,9,sharey=True)

for i in range(len(N.T[0])):
    axes[i].plot(MEM_tree[i],RES_tree[i],color = 'g',label = 'Tree')
    axes[i].plot(MEM_mps[i],RES_mps[i],color = 'r',label = 'Mps')
    axes[i].set_title('N='+str(int(N.T[0][i])))

axes[-1].legend()
# for ax in axes.flat:
#     ax.set(xlabel='Nb Coef', ylabel='Fidelity')
for ax in axes.flat:
    ax.label_outer()

plt.tight_layout()

plt.show()

# print(N.T.shape)
# print(CHI.shape)
# print(MEM_tree.shape)
# print(RES_tree.shape)