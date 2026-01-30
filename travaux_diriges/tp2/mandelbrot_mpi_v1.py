from mpi4py import MPI
from mandelbrot import MandelbrotSet
import numpy as np
from PIL import Image
import matplotlib.cm
from time import time


width, height = 4096, 4096
mandelbrot_set = MandelbrotSet(max_iterations=1000, escape_radius=2)


def mandelbrot_sequential():
    """Version séquentielle pour comparaison"""
    scaleX = 3./width
    scaleY = 2.25/height
    
    deb = time()
    convergence = np.empty((width, height), dtype=np.double)
    for y in range(height):
        for x in range(width):
            c = complex(-2. + scaleX*x, -1.125 + scaleY * y)
            convergence[x, y] = mandelbrot_set.convergence(c, smooth=True)
    fin = time()
    return convergence, fin - deb


def mandelbrot_mpi_master_slave():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    scaleX = 3./width
    scaleY = 2.25/height
    
    if rank == 0:  # MAITRE
        deb_total = time()
        convergence = np.empty((width, height), dtype=np.double)
        
        # Distribuer le travail aux esclaves
        chunk_height = height // (size - 1) if size > 1 else height
        
        # Envoyer les tâches initiales
        for slave in range(1, size):
            start = (slave - 1) * chunk_height
            end = height if slave == size - 1 else slave * chunk_height
            comm.send({'start': start, 'end': end}, dest=slave, tag=0)
        
        # Recevoir les résultats
        for slave in range(1, size):
            result = comm.recv(source=slave, tag=1)
            start = result['start']
            end = result['end']
            convergence[:, start:end] = result['data']
        
        fin_total = time()
        temps_mpi = fin_total - deb_total
        
        # Calculer la version séquentielle pour le speedup
        _, temps_seq = mandelbrot_sequential()
        speedup = temps_seq / temps_mpi
        efficiency = speedup / (size - 1) if size > 1 else 1.0
        
        # Constitution de l'image résultante
        deb = time()
        image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence.T)*255))
        fin = time()
        
        print(f"\n=== RÉSULTATS ===")
        print(f"Temps séquentiel : {temps_seq:.4f}s")
        print(f"Temps MPI ({size-1} esclaves) : {temps_mpi:.4f}s")
        print(f"Speedup : {speedup:.2f}x")
        print(f"Efficacité : {efficiency:.2%}")
        print(f"Temps de constitution de l'image : {fin-deb:.4f}s")
        print(f"Image sauvegardée dans mandelbrot_mpi.png")
        
        image.save("mandelbrot_mpi.png")
        
    else:  # ESCLAVES
        while True:
            task = comm.recv(source=0, tag=0)
            
            if task is None:  # Signal de fin
                break
            
            start = task['start']
            end = task['end']
            local_height = end - start
            
            deb = time()
            local_image = np.empty((width, local_height), dtype=np.double)
            for y in range(local_height):
                for x in range(width):
                    c = complex(-2. + scaleX*x, -1.125 + scaleY * (y + start))
                    local_image[x, y] = mandelbrot_set.convergence(c, smooth=True)
            fin = time()
            
            print(f"Rang {rank} - Calcul lignes {start}-{end} en {fin-deb:.4f}s")

            
            # Renvoyer le résultat
            result = {'start': start, 'end': end, 'data': local_image}
            comm.send(result, dest=0, tag=1)


mandelbrot_mpi_master_slave()