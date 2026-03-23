import torch
import torch.nn as nn
from models.components.attention import MultiHeadSelfAttention
from models.components.interaction_adapter import InteractionAdapter


class SSDNBlock(nn.Module):
    """A lightweight SSDN block: sequence self-attention + struct update + cross interaction + gate

    Minimal, fast-to-run prototype intended as an MVP fusion layer.
    """
    def __init__(self, embed_dim, pair_dim, num_heads=4, cross_heads=4, dropout=0.0):
        super().__init__()
        self.seq_attn = MultiHeadSelfAttention(embed_dim, pair_dim, num_heads)
        self.seq_ln = nn.LayerNorm(embed_dim)
        self.struct_ln = nn.LayerNorm(pair_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.ff_ln = nn.LayerNorm(embed_dim)

        # small MLP to refine struct embeddings
        self.struct_mlp = nn.Sequential(nn.Linear(pair_dim, pair_dim), nn.ReLU(), nn.Linear(pair_dim, pair_dim))

        # cross interaction adapter (pooled cross-attention)
        self.cross_adapter_seq = InteractionAdapter(embed_dim, cross_heads, dropout)
        self.cross_adapter_struct = InteractionAdapter(pair_dim, cross_heads, dropout) if pair_dim == embed_dim else None
        # projections to align dims between seq and struct pooled representations
        self.struct_to_seq = nn.Linear(pair_dim, embed_dim)
        self.seq_to_struct = nn.Linear(embed_dim, pair_dim)

        # gating to dynamically balance interaction
        self.gate = nn.Linear(embed_dim + pair_dim, embed_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, attn_mask=None):
        # x: (B, L, E), struct_embed: either (B, L, P) or (B, L, L, P)
        # Ensure struct_embed is pairwise (B, L, L, P) for compatibility with attention
        if struct_embed.dim() == 3:
            # expand token-level pair features to pairwise by outer product-like interaction
            # result shape: (B, L, L, P)
            struct_embed = torch.einsum('bid,bjd->bijd', struct_embed, struct_embed)

        seq_out, struct_out, attn = self.seq_attn(x, struct_embed, attn_mask, key_padding_mask)
        x = x + seq_out
        struct_embed = struct_embed + struct_out

        # FFN
        residual = x
        x = self.seq_ln(x + self.ff(x))

        # struct MLP
        struct_embed = self.struct_ln(struct_embed + self.struct_mlp(struct_embed))

        # pooled cross interaction (lightweight)
        seq_pool = x.mean(dim=1, keepdim=True)  # (B,1,E)
        # struct_embed is pairwise (B, L, L, P): pool over both sequence axes
        struct_pool = struct_embed.mean(dim=(1, 2)).unsqueeze(1)  # (B,1,P)

        # project pooled struct into seq space for cross-attention
        struct_pool_proj = self.struct_to_seq(struct_pool)
        seq_upd = self.cross_adapter_seq(seq_pool, struct_pool_proj)
        if self.cross_adapter_struct is not None:
            # if struct adapter exists and expects pair_dim==embed_dim
            # project seq pooled into struct dim first
            seq_pool_proj = self.seq_to_struct(seq_pool)
            struct_upd = self.cross_adapter_struct(seq_pool_proj, struct_pool)
        else:
            # project seq_pool into pair space if dims differ
            struct_upd = self.seq_to_struct(seq_pool)

        # broadcast pooled updates back
        # normalize updates along pooled dimension if adapters returned multiple pooled tokens
        if seq_upd.dim() > 2:
            seq_upd = seq_upd.mean(dim=1, keepdim=True)
        if struct_upd.dim() > 2:
            struct_upd = struct_upd.mean(dim=1, keepdim=True)

        # broadcast pooled updates back to token / pairwise shapes safely using expand
        L1 = x.shape[1]
        x = x + seq_upd.expand(-1, L1, -1)
        # struct_embed is (B, L, L, P); expand struct_upd to match
        L = struct_embed.shape[1]
        struct_embed = struct_embed + struct_upd.view(struct_upd.shape[0], 1, 1, struct_upd.shape[-1]).expand(-1, L, L, -1)

        # gating: compute gate from pooled representations and apply to sequence channels
        gate = torch.sigmoid(self.gate(torch.cat([seq_pool.squeeze(1), struct_pool.squeeze(1)], dim=-1))).unsqueeze(1)
        x = (1 - gate) * residual + gate * x

        return x, struct_embed, attn


class SSDN(nn.Module):
    def __init__(self, embed_dim, pair_dim, num_layers=2, num_heads=4, cross_heads=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SSDNBlock(embed_dim, pair_dim, num_heads=num_heads, cross_heads=cross_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.pair_final_layer_norm = nn.LayerNorm(pair_dim)

    def forward(self, x, struct_embed, key_padding_mask=None, need_attn_weights=False, attn_mask=None):
        attn_weights = [] if need_attn_weights else None
        for block in self.blocks:
            x, struct_embed, attn = block(x, struct_embed, key_padding_mask, attn_mask)
            if need_attn_weights:
                attn_weights.append(attn)

        x = self.final_layer_norm(x)
        struct_embed = self.pair_final_layer_norm(struct_embed)
        return x, struct_embed, attn_weights
