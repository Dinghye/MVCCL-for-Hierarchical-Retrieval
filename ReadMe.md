# MVCCL for Hierarchical Retrieval

A hierarchical retrieval prototype based on WordNet subtrees, including the Regular baseline, HR pre-training + long-distance fine-tuning, and three-stage MVCCL training.

## Env
```bash
pip install torch torchvision torchaudio
pip install nltk networkx sentence-transformers matplotlib
python -m nltk.downloader wordnet
```

The first run will automatically download the SBERT model (all-MiniLM-L6-v2).

## Quick Start
```bash
# By default, the WordNet subtree is constructed with "animal.n.01" as the root.
python main.py
```
After running, it will print the Recall@K values for each distance and generate a comparison curve.`recall_by_distance.png`

Current Result:
| Structural Distance(sample num N)| MVCCL  | HR Pretrain-Finetune | Regular |   |   |   |   |   |   |
|-----------------------------|--------|----------------------|---------|---|---|---|---|---|---|
| 0 (n=3065)                  | 11.81% | 70.83%               | 100.00% |   |   |   |   |   |   |
| 1 (n=3080)                  | 19.81% | 39.09%               | 65.71%  |   |   |   |   |   |   |
| 2 (n=3085)                  | 45.12% | 44.83%               | 39.94%  |   |   |   |   |   |   |
| 3 (n=3068)                  | 70.40% | 63.53%               | 19.88%  |   |   |   |   |   |   |
| 4 (n=3007)                  | 88.16% | 33.56%               | 7.12%   |   |   |   |   |   |   |
| 5 (n=2809)                  | 97.37% | 12.25%               | 2.28%   |   |   |   |   |   |   |
| 6 (n=2337)                  | 99.36% | 3.04%                | 0.34%   |   |   |   |   |   |   |
| 7 (n=1694)                  | 98.76% | 1.00%                | 0.18%   |   |   |   |   |   |   |
| OVERALL (n=22145)           | 62.79% | 36.79%               | 32.60%  |   |   |   |   |   |   |


Commonly adjustable parameters in `config.py`:
- Tree depth, ancestor distance: `MAX_DEPTH`, `MAX_ANCESTOR_DIST`
- Long-distance threshold: `LONG_DIST_THR`
- Training configuration: `BATCH_SIZE`, `LR`, `PRETRAIN_EPOCHS`, `FINETUNE_EPOCHS`, `MVC_STAGES`
- Projection dimension/Device: `EMBED_DIM`, `DEVICE`

## What have been done?
- Build WordNet subtree: Traverse hyponyms starting from `ROOT_LEMMA` in a downward manner, with the edge direction set as child→parent. Generate ancestor/distance and parent/brother mappings.
- Text and base vector: Combine the lemma+definition of the synset into a text, encode it into a fixed vector using SBERT, and calculate the semantic neighbor list.
- Model: Dual Tower `DualEncoder`, with fixed SBERT vectors, only training the **linear projections** of query/doc.
- Training three lines:
  - Regular: in-batch InfoNCE, with any ancestor as the positive sample.
  - HR: Start from the Regular weights, only using long-distance positive samples for fine-tuning.
  - MVCCL: Three-stage curriculum learning, gradually increasing the difficulty of negative samples (random → sibling nodes → semantic neighbors but with a different structure).
- Evaluation: Bucket the recall@K based on structural distance and draw a comparison curve.


## Code Structure
- `main.py`: The complete pipeline entry point (graph construction, coding, three types of training, evaluation and plotting).
- `config.py`: Hyperparameters and device/random seed management.
- `wordnet_utils.py`: Subtree construction, ID assignment, ancestors/distance and parent/child/brother relationships.
- `embedding_utils.py`: SBERT encoding and semantic neighbor calculation.
- `modeling.py`: Definition of the dual-tower model;
- `losses.py`: InfoNCE/MVCCL losses.
- `datasets.py`: Three types of datasets: Regular, LongDistance, MvCcl.
- `training.py`: Three training loops;
- `evaluation.py`: Recall@K evaluation and plotting.


## TODO List
1. (Optional) Currently, there is a catastrophic forgetting issue in MVCCL, where the farther the node, the higher the accuracy. Consider addressing this problem.
2. (Optional) Add larger and more datasets for experiments. For example, we can use large root subtrees of types such as `organism.n.01`, `object.n.01`, `physical_entity.n.01` (6k - 12k nodes), or we can add datasets like FAISS (50k - 200k nodes).
3. Final document and slide preparation