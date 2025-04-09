
import quimb as qu
import quimb.tensor as qtn

# 🔹 Étape 1: Créer un état de Bell |Ψ-⟩ = (|01⟩ - |10⟩) / √2
bell_psi_minus = qu.bell_state('psi-')

# 🔹 Étape 2: Construire la matrice densité ρ_AB = |Ψ-⟩⟨Ψ-|
rho_AB = bell_psi_minus @ bell_psi_minus.H  # Produit extérieur

# 🔹 Étape 3: Effectuer la trace partielle sur le qubit B pour obtenir ρ_A
rho_A = qu.partial_trace(rho_AB, dims=[2, 2], keep=[0])  # Garde le premier qubit
entropyA= qu.entropy_subsys(bell_psi_minus, dims= [2,2], sysa=[0])

print("ENTROPY A EST", entropyA)

# 🔹 Étape 4: Calculer l'entropie de von Neumann S(ρ_A)
entropy_A = qu.entropy(rho_A)

# 🔹 Affichage des résultats
print("Matrice densité totale ρ_AB :\n", rho_AB)
print("\nMatrice densité réduite ρ_A :\n", rho_A)
print("\nEntropie d'intrication (S(ρ_A)) :", entropy_A)

