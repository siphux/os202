from mpi4py import MPI
from mandelbrot import MandelbrotSet
import numpy as np
from PIL import Image
import matplotlib.cm
from time import time


def mandelbrot_mpi_cyclique():    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
    width, height = 2048, 2048
    scaleX = 3./width
    scaleY = 2.25/height
    
    
    
    if rank == 0:
        deb_total = time()
    

    convergence_local = {}
    deb = time()
    for y in range(height):
        if y%size == rank:
            convergence_row = np.empty(width, dtype=np.double)
            for x in range(width):
                c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
                convergence_row[x] = mandelbrot_set.convergence(c, smooth = True)
            convergence_local[y] = convergence_row
    fin = time()
    
    print(f"Processus {rank} a calculé {len(convergence_local)} lignes en {fin-deb:.4f} secondes.")
    
    # Regroupement avec gather (minuscule) + vstack
    all_local = comm.gather(convergence_local, root=0)
    
    if rank == 0:
        convergence = np.empty((height, width), dtype=np.double)
        for local_dict in all_local:
            for y, row in local_dict.items():
                convergence[y, :] = row
        
        fin_total = time()
        print(f"Temps total du calcul de l'ensemble de Mandelbrot : {fin_total - deb_total:.4f} secondes.")
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence)*255))
        fin = time()
        print(f"Temps de constitution de l'image : {fin-deb:.4f} secondes.")
        image.save("mandelbrot_mpi_equitable.png")
        print("Image sauvegardée dans mandelbrot_mpi_equitable.png")

if __name__ == "__main__":
    mandelbrot_mpi_cyclique()