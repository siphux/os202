from mpi4py import MPI
from mandelbrot import MandelbrotSet
import numpy as np
from PIL import Image
import matplotlib.cm
from time import time


def mandelbrot_mpi_equidistribue():
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 2048, 2048
    scaleX = 3./width
    scaleY = 2.25/height
    
    height_per_process = height // size
    remainder = height % size
    start_y = rank * height_per_process + min(rank, remainder)
    end_y = start_y + height_per_process + (1 if rank < remainder else 0)
    local_height = end_y - start_y
    
    if rank == 0:
        deb_total = time()
    
    convergence_local = np.empty((local_height, width), dtype=np.double)
    deb = time()
    for y in range(local_height):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * (start_y + y))
            convergence_local[y, x] = mandelbrot_set.convergence(c, smooth=True)
    fin = time()
    
    print(f"Processus {rank} a calculé de y={start_y} à y={end_y} en {fin-deb:.4f} secondes.")
    
    comm.Barrier()
    
    if rank == 0:
        # Processus 0 crée le tableau final et place ses données
        convergence = np.empty((height, width), dtype=np.double)
        convergence[start_y:end_y, :] = convergence_local
        
        # Recevoir des autres processus
        for source in range(1, size):
            other_start_y = source * height_per_process + min(source, remainder)
            other_end_y = other_start_y + height_per_process + (1 if source < remainder else 0)
            other_local_height = other_end_y - other_start_y
            
            buffer = np.empty((other_local_height, width), dtype=np.double)
            comm.Recv(buffer, source=source)
            convergence[other_start_y:other_end_y, :] = buffer
        
        # Seul le processus 0 affiche les temps finaux
        fin_total = time()
        print(f"Temps total du calcul de l'ensemble de Mandelbrot : {fin_total - deb_total:.4f} secondes.")
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin-deb:.4f} secondes.")
        image.save("mandelbrot_mpi_equitable.png")
        print("Image sauvegardée dans mandelbrot_mpi_equitable.png")
    else:
        # Les autres processus envoient vers le processus 0
        comm.Send(convergence_local, dest=0)


def mandelbrot_mpi_equidistribue_gather():
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 2048, 2048
    scaleX = 3./width
    scaleY = 2.25/height
    
    height_per_process = height // size
    remainder = height % size
    start_y = rank * height_per_process + min(rank, remainder)
    end_y = start_y + height_per_process + (1 if rank < remainder else 0)
    local_height = end_y - start_y
    
    if rank == 0:
        deb_total = time()
    
    convergence_local = np.empty((local_height, width), dtype=np.double)
    deb = time()
    for y in range(local_height):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * (start_y + y))
            convergence_local[y, x] = mandelbrot_set.convergence(c, smooth=True)
    fin = time()
    
    print(f"Processus {rank} a calculé de y={start_y} à y={end_y} en {fin-deb:.4f} secondes.")
    
    # Regroupement avec gather (minuscule) + vstack
    all_local = comm.gather(convergence_local, root=0)
    
    if rank == 0:
        # Stack les tableaux verticalement
        convergence = np.vstack(all_local)
        
        fin_total = time()
        print(f"Temps total du calcul de l'ensemble de Mandelbrot : {fin_total - deb_total:.4f} secondes.")
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin-deb:.4f} secondes.")
        image.save("mandelbrot_mpi_equitable.png")
        print("Image sauvegardée dans mandelbrot_mpi_equitable.png")

if __name__ == "__main__":
    
    mandelbrot_mpi_equidistribue_gather() # temps equivalents (celui executé en second est toujours légèrement plus rapide)
    # mandelbrot_mpi_equidistribue()
    
    