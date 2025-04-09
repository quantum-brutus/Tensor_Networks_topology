import quimb.tensor as qtn
import numpy as np
import random


class FullyConnectedGraphTensorGenerator:
    def __init__(self, L=15, min_children=0, max_children=3, phys_dim=2, bond_size_rule="exponential", leaf_probability=0.3, seed=None):
        """
        Générateur de graphes connectés de tenseurs avec des feuilles possibles.

        - L : Nombre total de nœuds
        - min_children : Nombre minimum d'enfants par nœud (0 = possibilité de feuille)
        - max_children : Nombre maximum d'enfants par nœud
        - phys_dim : Dimension physique des tenseurs
        - bond_size_rule : Règle de taille des liens ("fixed", "exponential", "logarithmic")
        - leaf_probability : Probabilité qu'un nœud soit une feuille (ajoutée après la connexion principale)
        - seed : Valeur pour fixer le générateur aléatoire (assure la reproductibilité)
        """
        self.L = L
        self.min_children = min_children
        self.max_children = max_children
        self.phys_dim = phys_dim
        self.bond_size_rule = bond_size_rule
        self.leaf_probability = leaf_probability
        self.seed = seed

        # Fixer la seed pour la reproductibilité
        self.set_seed()

        # Générer l'arbre avec la seed fixée
        self.children_map = self.generate_fully_connected_tree()

    def set_seed(self):
        """Fixe la seed des générateurs aléatoires pour assurer la reproductibilité."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate_fully_connected_tree(self):
        """Génère un arbre couvrant (spanning tree) assurant la connectivité."""
        children_map = {i: [] for i in range(self.L)}
        available_nodes = list(range(1, self.L))  # Noeuds disponibles pour être enfants
        used_nodes = {0}  # On commence avec la racine

        # Construction d'un arbre couvrant pour assurer la connectivité
        while available_nodes:
            parent = random.choice(list(used_nodes))
            child = available_nodes.pop(0)
            children_map[parent].append(child)
            used_nodes.add(child)

        # Ajout aléatoire d'autres connexions pour diversifier la structure
        for i in range(self.L):
            if random.random() < self.leaf_probability and i != 0:
                continue  # Ce nœud devient une feuille (on ne lui ajoute pas d'enfants)

            num_children = random.randint(self.min_children, self.max_children)
            possible_children = [n for n in range(self.L) if n not in children_map[i] and n != i]

            random.shuffle(possible_children)
            selected_children = possible_children[:num_children]

            for child in selected_children:
                if child not in used_nodes:
                    children_map[i].append(child)
                    used_nodes.add(child)

        return children_map

    def compute_bond_size(self, depth):
        """Calcule la taille des liens en fonction de la profondeur et de la règle choisie."""
        if self.bond_size_rule == "fixed":
            return 2  # Taille constante pour tous les liens
        elif self.bond_size_rule == "exponential":
            return 2 ** (depth + 1)  # Croissance exponentielle
        elif self.bond_size_rule == "logarithmic":
            return max(2, int(np.log2(depth + 2)))  # Croissance logarithmique
        else:
            raise ValueError("Règle de taille de lien inconnue ! Utilise 'fixed', 'exponential' ou 'logarithmic'.")

    def generate_graph(self):
        """Construit un graphe totalement connecté en respectant la structure générée."""
        tensors = [qtn.Tensor(data=1.0) for _ in range(self.L)]

        for i in range(self.L):
            # Ajouter un indice physique
            tensors[i].new_ind(f'k{i}', size=self.phys_dim)

            # Déterminer la profondeur du nœud
            depth = int(np.floor(np.log2(i + 1)))

            # Connexion avec les enfants définis dans children_map
            for child in self.children_map.get(i, []):
                bond_size = self.compute_bond_size(depth)
                tensors[i].new_bond(tensors[child], size=bond_size)

        return qtn.TensorNetwork(tensors)


