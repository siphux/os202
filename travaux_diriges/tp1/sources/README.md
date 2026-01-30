
# TD1

`pandoc -s --toc README.md --css=./github-pandoc.css -o README.html`

## lscpu

*lscpu donne des infos utiles sur le processeur : nb core, taille de cache :*

```
Coller ici les infos *utiles* de lscpu.
```


## Produit matrice-matrice

### Effet de la taille de la matrice

  n            | MFlops       | CPU time (s)
---------------|--------------|--------------
1024 (origine) | 66.4885      | 32.2986
1023           | 693.084      | 3.08938
1025           | 545.95       | 3.9


*Expliquer les résultats.*
Surprenant, la complexité est 2n + 2n*2, mais temps(1024) > temps(1025)
On observe un effet néfaste de la mémoire : le cache étant organisé en ligne de cache (tag + data (8 doubles)). Si on a un multiple de 2, on va faire du cache-flushing : on va remplir les lignes de cache puis les reremplir, c'est le pire cas possible. La bande-passante mémoire est le bottleneck et non pas les puissances de calcul CPU.
### Permutation des boucles

*Expliquer comment est compilé le code (ligne de make ou de gcc) : on aura besoin de savoir l'optim, les paramètres, etc. Par exemple :*

`make TestProduct.exe && ./TestProduct.exe 1024`


  ordre           | time(s)    | MFlops  | MFlops(n=2048)
------------------|---------|---------|----------------
i,j,k (origine)   | 12.2    | 176.4   |
j,i,k             | 14.1    | 151.8  |
i,k,j             | 32.3    | 66.5    |
k,i,j             | 38.9    | 55.1   |
j,k,i             | 0.66    | 3241.8  |
k,j,i             | 0.74    | 2873.8 |


*Discuter les résultats.*
Deux configurations donnent des résultats nettement supérieurs, (j, k, i) et (k, j, i). Dans ces deux configurations l'indice i est le plus interne, le plus à la fin. C'est de nouveau une histoire de cache. Cela vient de la définition des matrices, où le tableau stocke les éléments de haut en bas, donc i constant pour j qui varie, pour chaque colonne. Boucler sur i en dernier, permet de stocker colonne par colonne. On a des accès mémoire contigus.


### OMP sur la meilleure boucle

`make TestProduct.exe && OMP_NUM_THREADS=8 ./TestProduct.exe 1024`

Si #pragma omp parallel for avant la boucle en i.

  OMP_NUM         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
1                 | 1655.46
2                 | 1480.6
3                 | 1385.8
4                 | 1281.1
5                 | 1412.8
6                 | 1279.17
7                 | 1286.4
8                 | 1287


Si #pragma omp parallel for avant toutes les boucles

  OMP_NUM         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
1                 | 2450.21
2                 | 3865.6
3                 | 5154.61
4                 |6391.12
5                 | 7850.89
6                 | 8658.66
7                 | 9607.88
8                 | 9576.12
9                 | 10810
10                 | 10290.1
11                 | 8106.77
12                | 11253
13                | 11021.9
14                | 10435.8
15                | 11263.4
16                | 10276.8

*Tracer les courbes de speedup (pour chaque valeur de n), discuter les résultats.*
Scaling idéal S_ideal = N_proc (on aimerait avoir une courbe affine)





### Produit par blocs

`make TestProduct.exe && ./TestProduct.exe 1024`



  szBlock         | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)
------------------|---------|----------------|----------------|---------------
origine (=max)    |
32                |
64                |
128               |
256               |
512               |
1024              |

*Discuter les résultats.*



### Bloc + OMP


  szBlock      | OMP_NUM | MFlops  | MFlops(n=2048) | MFlops(n=512)  | MFlops(n=4096)|
---------------|---------|---------|----------------|----------------|---------------|
1024           |  1      |         |                |                |               |
1024           |  8      |         |                |                |               |
512            |  1      |         |                |                |               |
512            |  8      |         |                |                |               |

*Discuter les résultats.*


### Comparaison avec BLAS, Eigen et numpy

cf tel pour valeurs

*Comparer les performances avec un calcul similaire utilisant les bibliothèques d'algèbre linéaire BLAS, Eigen et/ou numpy.*


# Tips

```
	env
	OMP_NUM_THREADS=4 ./produitMatriceMatrice.exe
```

```
    $ for i in $(seq 1 4); do elap=$(OMP_NUM_THREADS=$i ./TestProductOmp.exe|grep "Temps CPU"|cut -d " " -f 7); echo -e "$i\t$elap"; done > timers.out
```



#2.1

Néanmoins un problème avec l'importation de la librarie mpi4py m'empeche de pouvoir tester les programmes...

from mpi4py import MPI
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    next_rank = (rank + 1) % size
    prev_rank = (rank - 1 + size) % size

    token = None

    if rank == 0:
        token = 1
        print(f"{rank} initialise le jeton à {token} et l'envoie à {next_rank}")
        
        comm.send(token, dest=next_rank)
        
        token = comm.recv(source=prev_rank)
        print(f"{rank} a obtenu le jeton final obtenu. Valeur = {token}")


    else:
        token = comm.recv(source=prev_rank)
        print(f"{rank} a obtenu le jeton. Valeur = {token}")
        token += 1
        comm.send(token, dest=next_rank)
        print(f"{rank} a envoyé le jeton. Valeur = {token}")

if __name__ == "__main__":
    main()


2.2
