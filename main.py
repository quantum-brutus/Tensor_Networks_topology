from generate_trees import *


seeds = 30 ## A CHANGER
length = 25
leaf_proba = 1

# Exemples d'utilisation :
if __name__ == "__main__":
    print("Génération de graphes totalement connectés avec seed.\n")

    # Graphes reproductibles avec seed fixée
    graph1 = FullyConnectedGraphTensorGenerator(L=length, min_children=2, max_children=5, bond_size_rule="exponential", leaf_probability=leaf_proba, seed=seeds).generate_graph()
    graph1.draw(title=f"Liens exponentiels (seed={seeds}, L={length}, leaf probability ={leaf_proba})", show_inds=True)

    graph2 = FullyConnectedGraphTensorGenerator(L=length, min_children=0, max_children=3, bond_size_rule="fixed", leaf_probability=leaf_proba, seed=seeds).generate_graph()
    graph2.draw(title=f"Liens fixes (seed={seeds}, L={length}, leaf probability ={leaf_proba})", show_inds=True)

    print("Graphes générés avec succès")
