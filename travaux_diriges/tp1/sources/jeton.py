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