import matplotlib.pyplot as plt
import numpy as np

# Some utils
class PlotUtils():
    def __init__(self):
        pass

    def plot_W(self, weights):
        weights = np.around(np.array(weights), decimals=2)
        fig, ax = plt.subplots(figsize=(7, 7))
        hm = ax.imshow(weights, cmap='inferno')
        for i in range(len(weights)):
            for j in range(len(weights)):
                weight = ax.text(j, i, weights[i, j], 
                                ha="center", va="center", color="b", fontsize=5)

        fig.tight_layout()
        plt.show()

    def plot_matrix(self, matrix, name):
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap='inferno')
        plt.title(f"Matrix {name}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()