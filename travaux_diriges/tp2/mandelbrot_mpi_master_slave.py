from mpi4py import MPI
from mandelbrot import MandelbrotSet
import numpy as np
from PIL import Image
import matplotlib.cm
from time import time
import os


def mandelbrot_mpi_master_slave():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 2048, 2048
    scaleX = 3./width
    scaleY = 2.25/height

    if rank == 0:
        deb_total = time()
        convergence = np.empty((height, width), dtype=np.double)
        indexes_to_compute = [i for i in range(height)]
        next_idx = 0
        for slave in range(1, size):
            if next_idx < height:
                comm.send(indexes_to_compute[next_idx], dest = slave)
                next_idx += 1
            else:
                comm.send(-1, dest = slave)
        
        lignes_recues = 0
        while lignes_recues < height:
            status = MPI.Status()
            data = comm.recv(source=MPI.ANY_SOURCE, status=status)
            source = status.Get_source()
            y, row = data
            convergence[y, :] = row
            lignes_recues += 1
            if next_idx < height:
                comm.send(indexes_to_compute[next_idx], dest = source)
                next_idx += 1
            else:
                comm.send(-1, dest = source)

        fin_total = time()
        print(f"Temps total du calcul de l'ensemble de Mandelbrot : {fin_total - deb_total:.4f} secondes.")
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin-deb:.4f} secondes.")
        image.save("mandelbrot_mpi_master_slave.png")
        print("Image sauvegardée dans mandelbrot_mpi_master_slave.png")
    else:
        while True:
            y = comm.recv(source=0)
            if y == -1:
                break
            row = np.empty(width, dtype=np.double)
            for x in range(width):
                c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
                row[x] = mandelbrot_set.convergence(c, smooth=True)
            comm.send((y, row), dest=0)

if __name__ == "__main__":
    mandelbrot_mpi_master_slave()
