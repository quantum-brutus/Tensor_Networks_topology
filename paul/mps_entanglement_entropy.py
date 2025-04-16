import quimb as qu
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Paramètres du système
L = 6  # Nombre de qubits
bond_dims = [1, 2, 16, 128]  # Différentes bond dimensions à tester
seed = 42  # Seed pour la reproductibilité

# 🔹 Stocker les entropies pour chaque bond dimension
entropies_per_bond = {bd: [] for bd in bond_dims}

# 🔹 Boucle sur chaque bond dimension
for bond_dim in bond_dims:
    mps = qtn.MPS_rand_state(L, bond_dim=bond_dim, seed=seed)  # Génération du MPS

    entropies = []  # Stocker l'entropie pour chaque qubit
    psi = mps.to_dense()  # Conversion du MPS en état global

    for i in range(L):
        S_i = qu.entropy_subsys(psi, dims=[2] * L, sysa=[i])  # Calcul de l'entropie
        entropies.append(S_i)

    entropies_per_bond[bond_dim] = entropies  # Sauvegarde des résultats

# 🔹 Affichage des résultats sous forme de graphique
plt.figure(figsize=(10, 6))

for bond_dim, entropies in entropies_per_bond.items():
    plt.plot(range(L), entropies, marker='.', linestyle='-', label=f"Bond Dim {bond_dim}")

plt.xlabel("Position de la coupe")
plt.ylabel("Entropie d'intrication")
plt.title("Évolution de l'entropie d'intrication en fonction de la bond dimension et de la position de la partition dans un MPS aléatoire donné.")
plt.legend()
plt.grid()
plt.show()

# 🔹 Affichage des valeurs d'entropie
for bond_dim, entropies in entropies_per_bond.items():
    print(f"\nBond Dimension {bond_dim}:")
    for i, S_i in enumerate(entropies):
        print(f"Qubit {i} - Entropie d'intrication : {S_i:.4f}")
