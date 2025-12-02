import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def encode_texts_with_sbert(texts, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
    """Encode the text into fixed embeddings using Sentence-Transformer."""
    sbert = SentenceTransformer(model_name, device=device)
    embs = sbert.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return torch.tensor(embs, dtype=torch.float32)


def compute_semantic_neighbors(base_embs: torch.Tensor, top_k: int):
    """
    Calculate the semantic neighbor list for each node (sorted by SBERT cosine similarity).
    Return sem_neighbors[i] = [j1, j2, ..., jk].
    """
    with torch.no_grad():
        embs = F.normalize(base_embs, p=2, dim=-1)
        sims = embs @ embs.t()
        num_nodes = embs.size(0)
        sem_neighbors = []

        for i in range(num_nodes):
            row = sims[i]
            _, idx = torch.sort(row, descending=True)
            idx = [j for j in idx.tolist() if j != i]
            sem_neighbors.append(idx[:top_k])

    return sem_neighbors
