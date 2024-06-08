import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import PlotUtils

np.random.seed(42)
torch.manual_seed(42)
plotter = PlotUtils()

# --------------------------------------------------------
# MAIN PARAMETERS AND DIMENSIONS
# --------------------------------------------------------
# The original implementation uses same dimensions for d and v for practical reasons
# In the base model of Vaswani et al. (2017), both k and v are 64
# This is because they use 512 dimension for the model (embedding) 
# and paralelize the computations in 8 heads: 512 // 8 = 64/head
d_model = 512
heads = 8
dim_k = d_model // heads
dim_v = d_model // heads

# --------------------------------------------------------
# MAKE TOY EMBEDDINGS
# --------------------------------------------------------
# sequence lenght; how many "words" to analyze; kind of block-size
n_embeddings = 128 
# Make some random embedding vectors. dim_k is because we assume
# that this is 1 of 8 attention heads
embeddings = torch.randn(n_embeddings, dim_k)


# --------------------------------------------------------
# ADDING POSITIONAL ENCODING
# --------------------------------------------------------
def positional_encoding(d_model, seq_len):
    # make a matrix of zeroes to fill
    pe = torch.zeros(seq_len, d_model)
    # get total positions
    position = torch.arange(0, seq_len).unsqueeze(1)
    # get the division terms for the sequence
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10_000) / d_model)))
    # Apply sine and cosine to even and odd positions filling the initial matrix of zeroes
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    # return the positional embeddings
    return pe

# positions to 64 to make it match with head vectors
positional = positional_encoding(d_model=dim_k, seq_len=n_embeddings)
# add positional information to embeddings
embeddings = embeddings + positional

plotter.plot_matrix(positional, 'Positional Embeddings')

# --------------------------------------------------------
# WEIGHT MATRICES
# --------------------------------------------------------
# make the linear layer matrices to be multiplied by embeddings
# We don't need the bias terms, so we set it to False
W_k = nn.Linear(dim_k, dim_v, bias=False)
W_q = nn.Linear(dim_k, dim_v, bias=False)
W_v = nn.Linear(dim_k, dim_v, bias=False)

# Optional: to plot random weights
# plotter.plot_W(W_k.weight.data)
# plotter.plot_W(W_q.weight.data)
# plotter.plot_W(W_v.weight.data)

# --------------------------------------------------------
# APPLY Q, K, AND V TO W
# --------------------------------------------------------
# Project (i.e., apply) the embeddings to the weight matrices
# Q: Represents the vector to look up relevant information ("earch term")
# K: Each piece of information in the input ("identifier")
# V: The actual data or information that will be retrieved (content of K)

Q = W_q(embeddings)
K = W_k(embeddings)
V = W_v(embeddings)

# plot the matrices
grp = [Q, K, V]
names = ['QW Linear Projection', 'KW Linear Projection', 'VW Linear Projection']
for i, j in zip(grp, names):
    plotter.plot_matrix(i.detach().numpy(), j)

# --------------------------------------------------------
# ATTENTION MASK
# --------------------------------------------------------
# We need to prevent tokens from paying attention to subsequent
# words. We create a triangular matrix to mask those.
def token_masking(n):
    mask = torch.triu(torch.ones(n, n), diagonal=1)
    masked_seq = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return masked_seq

masking = token_masking(n_embeddings)

# --------------------------------------------------------
# GET ATTENTION
# --------------------------------------------------------
# Use Scaled Dot-Product Attention -> Attention(Q, K, T) = softmax(QK^T / sqrt(d_k))V
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim_k)
attention_scores = attention_scores + masking.unsqueeze(0) # apply masking
attention_weights = torch.softmax(attention_scores, dim=-1)
context_vector = torch.matmul(attention_weights, V)

# --------------------------------------------------------
# PLOT CONTEXT VECTOR AND ATTENTION WEIGHTS
# --------------------------------------------------------
# plot the masked attention scores: relevance or importance of each key to the corresponding query before applying the softmax
plotter.plot_matrix(attention_scores.squeeze().detach().numpy(), 'Masked Scores')
# plot the context vectors: encode the relevant information from the entire sequence for each query
plotter.plot_matrix(context_vector.squeeze().detach().numpy(), 'Context Vectors')
# Plot the self-attention: How much attention does each token put to other tokens in the sequence?
plotter.plot_matrix(attention_weights.squeeze().detach().numpy(), 'Self-Attention')

# --------------------------------------------------------
# APPLY LINEAR LAYER
# --------------------------------------------------------
# We make this small network. Note that, if using multihead, we would
# have concatenated all 8 heads' outputs to get back 512-dimension 
# vectors. However, here we resize the dimensions to 64 show the plot.

class PositionWiseFFNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFFNet, self).__init__()
        self.l_1 = nn.Linear(d_model, d_ff)
        self.l_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.l_2(self.dropout(torch.relu(self.l_1(x))))

ffn = PositionWiseFFNet(d_model=64, d_ff=2048, dropout=0.1)
out = ffn(context_vector)

plotter.plot_matrix(out.squeeze().detach().numpy(), 'Processed Output (input to next layer)')
