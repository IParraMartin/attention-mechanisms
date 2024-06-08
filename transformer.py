import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size # embedding size
        self.heads = heads # number of heads
        self.head_dim = embed_size // heads # dim of each attention head

        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by number of heads."

        # Linear layers to transform the inputs into K, Q, and V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # linear layer to concatenate the heads' outputs
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forwsrds(self, values, keys, queries, mask):
        N = queries.shape[0] # batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] # sequence lengths

        # split embedding into self.heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        """
        Energy = batched and multi-headed dot product operation

        The energy scores represent the similarity between queries and keys. 
        Higher scores indicate a closer match, which helps in identifying which 
        parts of the input are more relevant for generating the output.
        """
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, query_len, heads, head_dim)
        # energy shape: (N, query_len, heads, head_dim)

        """
        Ignore Irrelevant Positions: Masks help in ignoring positions that 
        are not relevant for attention (like padding tokens). This ensures 
        that the model does not attend to these positions, which could 
        otherwise skew the attention scores. It also helps the model learn 
        by 'hiding' future information.
        """
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # value shape: (N, value_len, heads, heads_dim)
        # after einsum: (N, query_len, heads, heads_dim) and then flatten last 2 dims

        out = self.fc_out(out)
        return out
