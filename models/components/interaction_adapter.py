import torch
import torch.nn as nn


class InteractionAdapter(nn.Module):
    """Lightweight cross-attention adapter for cross-encoder interaction.

    Usage: updated_query = adapter(query, ext_kv, query_mask=None, kv_mask=None)
    where query: (B, Q, E), ext_kv: (B, K, E)
    """
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, ext_kv, query_mask=None, kv_mask=None):
        # query: (B, Q, E), ext_kv: (B, K, E)
        # PyTorch MultiheadAttention expects key_padding_mask with True in positions that should be ignored
        attn_output, _ = self.attn(query, ext_kv, ext_kv, key_padding_mask=~kv_mask if kv_mask is not None else None,
                                   need_weights=False)
        query = query + attn_output
        query = self.layernorm(query)
        # simple FFN
        ff_out = self.ff(query)
        query = query + ff_out
        query = self.ff_norm(query)
        return query
