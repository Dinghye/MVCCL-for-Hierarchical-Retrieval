import random
import numpy as np
import torch

MAX_DEPTH = 7
MAX_ANCESTOR_DIST = 7
LONG_DIST_THR = 3
EMBED_DIM = 256
BATCH_SIZE = 128
LR = 1e-3
PRETRAIN_EPOCHS = 5
FINETUNE_EPOCHS = 9
TOP_K = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# MVCCL related
NEG_PER_SAMPLE = 5
MVC_STAGES = {
    "easy": 6,
    "medium": 4,
    "hard": 4,
}
SEM_TOPK = 50


def set_global_seeds(seed: int = SEED) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
