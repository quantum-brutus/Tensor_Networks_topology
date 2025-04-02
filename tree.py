import quimb as qu
import quimb.tensor as qtn
import numpy as np
import re
print('done')


class Tree:
    def __init__(self,root):
        self.nodes = {}
        self.network = None
        self.neutral = root
        self.root = root

        self.nodes[root.index] = root
        self.network = root.tensor

        ####### utile dans les autres fonc
    def to_str(self,node):
        return ''.join(node)

    def to_list(self,node):
        return re.findall(r'[A-Za-z]|\d+', node)

    def diff_part_str(self,A,B):
        A = self.to_list(A)
        B = self.to_list(B)

        prefix = []

        for i in range(min(len(A),len(B))):
            if A[i] == B[i]:
                prefix.append(A[i])
            else:
                break

        if not(prefix[-1].isnumeric()):
            prefix.pop()
        return prefix,A[len(prefix):],B[len(prefix):]

    def get_previous_node(self,node):
        nb = int(node[-1])
        if nb == 1:
            return node[0:-2]
        else:
            return node[0:-1]+[str(nb-1)]

    def is_adjacent(self,i,j):
        i = self.to_list(i)
        j = self.to_list(j)
        if self.get_previous_node(i) == j or self.get_previous_node(j) == i:
            return True
        else:
            return False

    def get_path_to_last_common_node(self,node,last_common_node):

        path = []
        
        previous_node = node
        while previous_node != last_common_node:
            path.append(self.to_str(previous_node))
            previous_node = self.get_previous_node(previous_node)

        path.append(self.to_str(last_common_node))
        return path

    def get_path(self,A,B):
        last_common_node,tail_A,tail_B = self.diff_part_str(A,B)

        A_to_lcn = list(reversed(self.get_path_to_last_common_node(self.to_list(A),last_common_node)))
        B_to_lcn = list(reversed(self.get_path_to_last_common_node(self.to_list(B),last_common_node)))


        for i in range(min(len(A_to_lcn),len(B_to_lcn))):
            if A_to_lcn[i] != B_to_lcn[i]:
                return list(reversed(A_to_lcn[i-1:]))+B_to_lcn[i:]
        return list(reversed(A_to_lcn[i+1:]))+B_to_lcn[i:]

    def swap_adjacent(self,i,j):
        # print('swapping ',i,j)
        array = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=float).reshape(2,2,2,2)
        self.apply_2qb_gate(i,j,gate_array = array)

    
    ###### methodes

    def shift_neutral(self,i,j,side = 'right'):

        # print('called shift_neutral on ',i,j)

        if not(self.is_adjacent(i,j)):
            raise Exception('not adjacent')
        
        # get nodes
        node_1 = self.nodes[i]
        node_2 = self.nodes[j]
        # get indexes
        side_1 = list(node_1.tensor.inds)
        side_2 = list(node_2.tensor.inds)
        # print('1',side_1)
        # print('2',side_2)
        # print('nodes',node_1.tensor,node_2.tensor)

        bond_ind = (set(node_1.tensor.inds) & set(node_2.tensor.inds)).pop()
        side_1.remove(bond_ind)
        side_2.remove(bond_ind)

        # print(bond_ind)
        # print(side_1)
        # print(side_2)

        # 'apply' gate
        contracted_qubits = node_1.tensor @ node_2.tensor

        split_tensor = contracted_qubits.split(left_inds=side_1,right_inds=side_2,method='svd',absorb=side,bond_ind=bond_ind)
        
        ind_1 = split_tensor.ind_map[side_1[0]].pop()
        tensor_1 = split_tensor.tensor_map[ind_1]
        ind_2 = split_tensor.ind_map[side_2[0]].pop()
        tensor_2 = split_tensor.tensor_map[ind_2]

        tensor_1.drop_tags([j])
        tensor_2.drop_tags([i])

        result = tensor_1 & tensor_2
        node_1.tensor = tensor_1
        node_2.tensor = tensor_2

        self.network.delete(i)
        self.network.delete(j)

        self.network = self.network & node_1.tensor
        self.network = self.network & node_2.tensor


    def shift_along_path(self,node,root):
        path = self.get_path(node,root)
        # print('shifting along ',path)
        
        for i in range(len(path)-1):
            self.shift_neutral(path[i],path[i+1],side = 'right')

    def add(self,node):
        self.nodes[node.index] = node

        if self.network == None:
            self.network = node.tensor
        else:
            self.network = self.network & node.tensor

        Id = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],dtype=float).reshape(2,2,2,2)
        self.apply_2qb_gate(i = node.index,j = node.up,gate_array=Id,already_linked=False)
        #canonalize
        self.shift_along_path(node.index,self.root.index)

    def apply_1qb_gate(self,i,gate_array = np.array([[1,0],[0,1]],dtype = complex)):
        node = self.nodes[i]
        gate = qtn.Tensor(data=gate_array,inds = ['out'+i,'nout'+i],tags = ['gate'])
        nqubit = self.network[i] & gate
        nqubit.contract_ind('out'+i)
        nqubit.reindex_({'nout'+i:'out'+i})
        nqubit.drop_tags('gate')

        node.tensor = nqubit[i]

        self.network.delete(i)

        self.network = self.network & nqubit

    def apply_2qb_gate(self,i,j,gate_array = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=float).reshape(2,2,2,2),side = 'right',already_linked=True):
        # print('called apply 2qb gate on ',i,j)
        # self.draw()
        swapped = False
        if not(self.is_adjacent(i,j)):
            # print('not adjacent,swapping')
            # raise Exception('not adjacent')
            path = self.get_path(i,j)
            for k in range(len(path)-2):
                self.swap_adjacent(path[k],path[k+1])
            old_i = i
            i = path[k+1]
            # print('--- swapping non adjacent over')
            swapped = True

        def sort_indexes(i,j):
            if len(i) > len(j):
                return i,j
            elif len(i) < len(j):
                return j,i
            elif int(i[-1]) > int(j[-1]):
                return i,j
            else:
                return j,i
        
        farthest_ind,closest_ind = sort_indexes(i,j)

        if already_linked:
            # print('shifting neutral to',i)
            self.shift_along_path(self.root.index,farthest_ind)
            # print('--- shifting neutral to ',i,' over')
            # self.draw()
            # print('before applying gate')
        
        # print('applying gate to',i,j)
        # get nodes
        node_1 = self.nodes[i]
        node_2 = self.nodes[j]

        #define gate
        gate = qtn.Tensor(data=gate_array,inds = ['out'+i,'out'+j,'nout'+i,'nout'+j],tags = ['gate'])

        if already_linked:
            bond_ind = j+'-'+i+'b'
        else:
            bond_ind = j+'-'+i

        #change indexes to match gate indexes

        gate = gate.split(left_inds=['out'+i,'nout'+i],method='svd',absorb='both',bond_ind=bond_ind)

        # 'apply' gate
        qubits = node_1.tensor & node_2.tensor
        qubits_and_gate = qubits & gate
        qubits_and_gate.contract_ind('out'+i)
        qubits_and_gate.contract_ind('out'+j)

        qu.tensor.tensor_fuse_squeeze(qubits_and_gate[i],qubits_and_gate[j])

        qubits_and_gate.reindex_({'nout'+i:'out'+i,'nout'+j:'out'+j})
        qubits_and_gate.drop_tags('gate')

        node_1.tensor = qubits_and_gate[i]
        node_2.tensor = qubits_and_gate[j]

        self.network.delete(i)
        self.network.delete(j)

        self.network = self.network & qubits_and_gate
        # print('gate applied on ',i,j)
        # self.draw()
        # print('after applying gate')

        if swapped:
            # print('swapping back')
            for k in range(1,len(path)-1):
                # print(k)
                self.swap_adjacent(path[:-1][-k],path[:-1][-k-1])
            # print('--- swapping back over')

        if already_linked:
            # print('putting neutral back to root from ',i)
            self.shift_along_path(farthest_ind,self.root.index)
        # print('draw after last shift')
        # self.draw()
        
    
    def draw(self):
        return self.network.draw(show_tags=True, show_inds='all',iterations=100, k=6)

    def swap(self,A,B):
        path = self.get_path(A,B)
        for i in range(len(path)-1):
            self.swap_adjacent(path[i],path[i+1])

class Node:
    def __init__(self,index,up,childrens,up_bdim,childrens_bdim):
        self.index = index
        self.up = up
        self.childrens = childrens
        self.up_bdim = up_bdim
        self.childrens_bdim = childrens_bdim
        self.init_tensor()
    
    def init_tensor(self):
        self.tensor = qtn.Tensor(data=np.array([1,0],dtype=complex),inds=['out'+self.index],tags = [self.index])

    def reindex(self,dic):
        self.tensor.reindex_(dic)

        for i in dic.keys():
            if self.up == i:
                self.up = dic[i]
            if i in self.childrens:
                self.childrens.remove(i)
                self.childrens.append(dic[i])


print('start tests')
for p in range(100):
    R0 = Node(index='R0',up = None,up_bdim = 0,childrens=['R0A1','R0B1'],childrens_bdim = [2,2])
    R0A1 = Node(index='R0A1',up = 'R0',up_bdim = 2,childrens=['R0A2'],childrens_bdim = [2])
    R0A2 = Node(index='R0A2',up = 'R0A1',up_bdim = 2,childrens=[],childrens_bdim = [])
    R0B1 = Node(index='R0B1',up = 'R0',up_bdim = 2,childrens=['R0B1A1','R0B1B1'],childrens_bdim = [2,2])
    R0B1A1 = Node(index='R0B1A1',up = 'R0B1',up_bdim = 2,childrens=[],childrens_bdim = [2])
    R0B1B1 = Node(index='R0B1B1',up = 'R0B1',up_bdim = 2,childrens=[],childrens_bdim = [2])

    Tree1 = Tree(R0)
    Tree1.add(R0A1)
    Tree1.add(R0A2)
    Tree1.add(R0B1)
    Tree1.add(R0B1A1)
    Tree1.add(R0B1B1)



    # Tree1.network.draw(show_tags=True, show_inds='all',iterations=100, k=4)

    print('***************** finished building tree **********************')

    # Tree1.swap_adjacent('R0A1','R0A2')
    # Tree1.apply_2qb_gate('R0A2','R0',gate_array = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=complex).reshape(2,2,2,2))
    # Tree1.shift_neutral('R0A1','R0A2',side = 'left')
    # Tree1.network.draw(show_tags=True, show_inds='all',iterations=100, k=4)


    translation_dic = {'R0':0,'R0A1':1,'R0A2':2,'R0B1':3,'R0B1A1':4,'R0B1B1':5}
    qubits = ['R0','R0A1','R0A2','R0B1','R0B1A1','R0B1B1']
    shuffled_qubits = qubits.copy()

    circ = qtn.Circuit(6)

    reindex_dic = {}
    for i in range(len(qubits)):
        reindex_dic['k'+str(i)] = 'out'+qubits[i]
    # print(reindex_dic)
    # net.draw(show_tags=True, show_inds='all',iterations=100, k=4)


    N_gates = 100
    ############# Gates
    # Tree1.apply_1qb_gate(i = 'R0',gate_array=np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]],dtype=complex))
    # circ.apply_gate('H',translation_dic['R0'])
    rng = np.random.default_rng()

    for i in range(N_gates):
        rng.shuffle(shuffled_qubits)
        if random() < 0.5:
            # print('Applied H to ', shuffled_qubits[0],translation_dic[shuffled_qubits[0]])
            Tree1.apply_1qb_gate(i = shuffled_qubits[0],gate_array=np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]],dtype=complex))
            circ.apply_gate('H',translation_dic[shuffled_qubits[0]])
        else:
            # print('Applied CNOT to ', shuffled_qubits[0],shuffled_qubits[1],translation_dic[shuffled_qubits[0]],translation_dic[shuffled_qubits[1]])
            Tree1.apply_2qb_gate(i = shuffled_qubits[0],j = shuffled_qubits[1],gate_array=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex).reshape(2,2,2,2))
            circ.apply_gate('CNOT',translation_dic[shuffled_qubits[0]],translation_dic[shuffled_qubits[1]])   
            

    #############

    net = circ.psi.reindex_(reindex_dic)

    ############

    fnet = net & Tree1.network
    # fnet.draw(show_tags=True, show_inds='all',iterations=100, k=4)
    print(fnet.contract())