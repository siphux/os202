import numpy as np
from mpi4py import MPI
from time import time


dim = 20000


def build_u(comm, rank):
    if rank == 0:
        u = np.arange(1, dim + 1, dtype=float)
    else:
        u = np.empty(dim, dtype=float)
    comm.Bcast(u, root=0)
    return u


def build_local_A_rowwise(start_row, end_row):
    rows = np.arange(start_row, end_row, dtype=int)[:, None]
    cols = np.arange(dim, dtype=int)[None, :]
    return ((rows + cols) % dim + 1.0).astype(float)


def build_local_A_colwise(start_col, end_col):
    rows = np.arange(dim, dtype=int)[:, None]
    cols = np.arange(start_col, end_col, dtype=int)[None, :]
    return ((rows + cols) % dim + 1.0).astype(float)


def mat_vec_parallel_colwise():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    u = build_u(comm, rank)
    total_time = time()
    num_col_per_proc = dim // size
    start_col = rank * num_col_per_proc
    end_col = (rank + 1) * num_col_per_proc if rank != size - 1 else dim
    local_A = build_local_A_colwise(start_col, end_col)
    local_v = local_A.dot(u[start_col:end_col])

    local_end = time()
    print(f"Processus {rank} a terminé en {local_end - total_time:.4f} secondes.")

    v = np.zeros(dim) if rank == 0 else None
    comm.Reduce(local_v, v, op=MPI.SUM, root=0)

    if rank == 0:
        total_end = time()
        print(f"Temps total du calcul parallèle : {total_end - total_time:.4f} secondes.")
        print(f"v = {v}")


def mat_vec_parallel_rowwise():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    u = build_u(comm, rank)
    total_time = time()
    num_rows_per_proc = dim // size
    start_row = rank * num_rows_per_proc
    end_row = (rank + 1) * num_rows_per_proc if rank != size - 1 else dim
    local_A = build_local_A_rowwise(start_row, end_row)
    local_v = local_A.dot(u)

    local_end = time()
    print(f"Processus {rank} a terminé en {local_end - total_time:.4f} secondes.")

    parts = comm.gather(local_v, root=0)

    if rank == 0:
        v = np.concatenate(parts)
        total_end = time()
        print(f"Temps total du calcul parallèle : {total_end - total_time:.4f} secondes.")
        print(f"v = {v}")


if __name__ == "__main__":
    mat_vec_parallel_rowwise()
    print("\n\n Rowwise \n\n")
    mat_vec_parallel_colwise()