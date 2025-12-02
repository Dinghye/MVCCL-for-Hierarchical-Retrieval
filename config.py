import random
import numpy as np
import torch


ROOT_LEMMA = "animal.n.01"

MAX_DEPTH = 5
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

EASY_REPLAY_RATIO_HARD = 0.1  # hard stage use 10% easy batch
TEMP_EASY = 0.07
TEMP_MEDIUM = 0.07
TEMP_HARD = 0.10  # hard tempo kinda higer to lower push

LAMBDA_RANK = 1.0  # The weight of the hierarchical sorting loss
RANK_MARGIN = 0.2  # ranking margin



def set_global_seeds(seed: int = SEED) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
