import torch
import matplotlib.pyplot as plt
torch.manual_seed(42)

tensor = torch.randn(10, 8)
modified = tensor.view(8, 10)

print(f'\nShape: {tensor.shape}\n\nOriginal Tensor: {tensor}\n\nNew Tensor: {modified}')

def plot_tensor(tensors=list(), main_name=str):
    data = [tensor.detach().numpy() for tensor in tensors]
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(data[0], cmap='plasma')
    ax[1].imshow(data[1], cmap='plasma')
    fig.suptitle(main_name, fontsize=20)
    plt.tight_layout()
    plt.show()

plot_tensor([tensor, modified], 'Before and after .view() method')


