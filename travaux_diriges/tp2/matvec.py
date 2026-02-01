# Produit matrice-vecteur v = A.u
import numpy as np
from mpi4py import MPI
from time import time 
# Dimension du problème (peut-être changé)
dim = 10_000
# Initialisation de la matrice

init_start = time()
A = np.array([[(i+j) % dim+1. for i in range(dim)] for j in range(dim)])
# print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i+1. for i in range(dim)])
# print(f"u = {u}")
init_end = time()
print(f"Initialisation en {init_end - init_start:.4f} secondes.")
# Produit matrice-vecteur

if __name__ == "__main__":

    print("\n\n Debut du calcul sequentiel \n\n")

    seq_start = time()
    v = A.dot(u)
    seq_end = time()
    print(f"Calcul sequentiel termine en {seq_end - seq_start:.4f} secondes.")

    print(f"v = {v}")




