import quimb as qu
import quimb.linalg.base_linalg as ql
import quimb.calc as qc
import numpy as np

# Définition de l'état de Bell |ψ⟩ = (|00⟩ + |11⟩) / √2
psi = qu.ket([1, 0, 0, 1]) / 2**0.5

print(psi)


# 2. Reshape en matrice 2x2 pour le bipartitionnement (qubit A | qubit B)
psi_mat = psi.reshape(2, 2)

print(psi_mat)

# 3. Décomposition SVD native de Quimb
U, S, Vh = ql.svd(psi_mat)

# 4. Les valeurs singulières S sont les coefficients de Schmidt
schmidt_coeffs = [i for i in S]

# 5. Calcul de l'entropie de Von Neumann avec quimb.calc.entropy
entropy_vn = qc.entropy(schmidt_coeffs)  

# 6. Affichage des résultats
print("Coefficients de Schmidt (valeurs singulières) :", schmidt_coeffs)
print("Entropie de Von Neumann :", entropy_vn)

