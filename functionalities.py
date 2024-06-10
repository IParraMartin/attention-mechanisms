import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

def plot_tensor(one_tensor=None, multiple_tensors=None, main_name=None):
    if multiple_tensors is not None:
        data = [tensor.detach().numpy() for tensor in multiple_tensors]
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(data[0], cmap='plasma')
        ax[1].imshow(data[1], cmap='plasma')
        fig.suptitle(main_name, fontsize=14)
    elif one_tensor is not None:
        data = one_tensor.detach().numpy()
        plt.imshow(data, cmap='plasma')
        plt.title(main_name, fontsize=14)
    else:
        print('No tensors were provided.')
        return None
    plt.tight_layout()
    plt.show()

heads = 8
model_d = 512
n_tokens = 10
k_dim = model_d // heads
v_dim = model_d // heads

linear = nn.Linear(k_dim, k_dim, bias=False)
print(f'\nShape of W: {linear.weight.size()}\n')

word_embeddings = torch.randn(n_tokens, k_dim)
print(f'Shape of d_model in head i: {word_embeddings.size()}\n')

projection = linear(word_embeddings)
print(f'Shape of W_x projection: {projection.size()}\n')

# plot_tensor(projection)
# plot_tensor(projection.view(64, 10))

"""
The resulting matrix C of the multiplication of matrices A and B 
has dimensions determined by the number of rows in A and the number 
of columns in B.
"""