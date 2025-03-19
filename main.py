from generate_trees import *


seeds = 42 ## A CHANGER

# Exemples d'utilisation :
if __name__ == "__main__":
    print("Génération de graphes totalement connectés avec seed.\n")

    # Graphes reproductibles avec seed fixée
    graph1 = FullyConnectedGraphTensorGenerator(L=25, min_children=0, max_children=3, bond_size_rule="exponential", leaf_probability=0.3, seed=seeds).generate_graph()
    graph1.draw(title=f"Graphe Connecté - Liens exponentiels (seed={seeds})", show_inds=True)

    graph2 = FullyConnectedGraphTensorGenerator(L=20, min_children=0, max_children=4, bond_size_rule="fixed", leaf_probability=0.5, seed=seeds).generate_graph()
    graph2.draw(title=f"Graphe Connecté - Liens fixes (seed={seeds})", show_inds=True)

    print("Graphes générés avec succès")
