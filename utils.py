import matplotlib.pyplot as plt
import numpy as np

# Some utils
class PlotUtils():
    def __init__(self):
        pass

    def plot_W(self, weights, name, color, xname=None, yname=None):
        weights = np.around(np.array(weights), decimals=2)
        plt.figure(figsize=(8, 8))
        plt.imshow(weights, cmap=color)

        if xname is not None:
            plt.xlabel(xname)
        if yname is not None:
            plt.ylabel(yname)

        plt.title(name, fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_matrix(self, matrix, name, color, xname=None, yname=None):
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap=color)

        if xname is not None:
            plt.xlabel(xname)
        if yname is not None:
            plt.ylabel(yname)
        
        plt.title(f"Matrix {name}", fontsize=14)
        plt.colorbar()
        plt.tight_layout()
        plt.show()