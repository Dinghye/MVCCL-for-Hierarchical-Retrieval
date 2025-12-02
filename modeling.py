import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoder(nn.Module):
    """
    Simple Dual Model
    - Fixed base text vector(eg SBERT), only train linear projections
    - The query and the doc each have a set of projection parameters (not shared)
    """
    def __init__(self, base_embeddings: torch.Tensor, proj_dim: int):
        super().__init__()
        num_nodes, base_dim = base_embeddings.shape
        self.register_buffer("base_embeddings", base_embeddings)

        self.query_proj = nn.Linear(base_dim, proj_dim)
        self.doc_proj = nn.Linear(base_dim, proj_dim)

    def encode_query(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embeddings[ids]
        z = self.query_proj(base)
        return F.normalize(z, p=2, dim=-1)

    def encode_doc(self, ids: torch.Tensor) -> torch.Tensor:
        base = self.base_embeddings[ids]
        z = self.doc_proj(base)
        return F.normalize(z, p=2, dim=-1)
