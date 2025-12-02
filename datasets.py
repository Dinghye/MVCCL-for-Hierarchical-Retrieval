import random
import torch
from torch.utils.data import Dataset

class RegularHrdataset(Dataset):
    """
    Regular Training: for each query q, random pick a ancestor as positive sample
    ancestors[q_id] is a ancestor id list (include itself, distance=0)
    """
    def __init__(self, all_q_ids, ancestors, distances):
        super().__init__()
        self.all_q_ids = all_q_ids
        self.ancestors = ancestors
        self.distances = distances

    def __len__(self):
        return len(self.all_q_ids)

    def __getitem__(self, idx):
        q_id = self.all_q_ids[idx]
        anc_list = self.ancestors[q_id]
        pos_id = random.choice(anc_list)
        dist = self.distances[(q_id, pos_id)]
        return q_id, pos_id, dist


class LongDistanceDataset(Dataset):
    """
    Finetune stage: only use 'long distance' as positive sample (q, anc), which is distance >= long_dist_thr
    """
    def __init__(self, all_q_ids, ancestors, distances, long_dist_thr: int):
        super().__init__()
        self.long_pairs = []
        for q_id in all_q_ids:
            for anc_id in ancestors[q_id]:
                dist = distances[(q_id, anc_id)]
                if dist >= long_dist_thr:
                    self.long_pairs.append((q_id, anc_id, dist))

        if len(self.long_pairs) == 0:
            raise ValueError("No long-distance sample, please lower LONG_DIST_THR or increase MAX_DEPTH。")

    def __len__(self):
        return len(self.long_pairs)

    def __getitem__(self, idx):
        q_id, pos_id, dist = self.long_pairs[idx]
        return q_id, pos_id, dist


class MvCclDataset(Dataset):
    """
    Multi-View Curriculum Contrastive Learning Dataset:
      stage in {"easy", "medium", "hard"}:
        - easy:   random nagative sample
        - medium: same parent's siblings as nagative sample (Structurally similar but with optional meanings)
        - hard:   emantic similar but structurally distant negative samples
    Positive sample uniformly referred to as ancestors(structural view), WE CAN EXTEND IT AS SEMANTIC POSTIVE SAMPLE HERE
    """
    def __init__(
        self,
        stage: str,
        all_q_ids,
        ancestors,
        distances,
        parents_of,
        siblings_of,
        sem_neighbors,
        num_nodes: int,
        neg_per_sample: int,
    ):
        super().__init__()
        assert stage in {"easy", "medium", "hard"}
        self.stage = stage
        self.all_q_ids = all_q_ids
        self.ancestors = ancestors
        self.distances = distances
        self.parents_of = parents_of
        self.siblings_of = siblings_of
        self.sem_neighbors = sem_neighbors
        self.num_nodes = num_nodes
        self.neg_per_sample = neg_per_sample
        self.all_ids_set = set(range(num_nodes))

    def __len__(self):
        return len(self.all_q_ids)

    def __getitem__(self, idx):
        q_id = self.all_q_ids[idx]
        pos_id = random.choice(self.ancestors[q_id])
        dist = self.distances[(q_id, pos_id)]

        if self.stage == "easy":
            neg_ids = self.sample_easy_negatives(q_id)
        elif self.stage == "medium":
            neg_ids = self.sample_medium_negatives(q_id)
        else:
            neg_ids = self.sample_hard_negatives(q_id)

        return q_id, pos_id, dist, torch.tensor(neg_ids, dtype=torch.long)

    # --------- 各阶段的负样本策略 ---------

    def sample_easy_negatives(self, q_id):
        """Easy: Random negative samples, avoid selecting ancestors."""
        avoid = set(self.ancestors[q_id])
        avoid.add(q_id)
        candidates = list(self.all_ids_set - avoid)
        if len(candidates) == 0:
            candidates = list(self.all_ids_set - {q_id})
        if len(candidates) <= self.neg_per_sample:
            return random.sample(candidates, len(candidates))
        return random.sample(candidates, self.neg_per_sample)

    def sample_medium_negatives(self, q_id):
        """
        Medium: 
        Use the siblings of the parent as negative samples.
        If the number of siblings is insufficient, then fill the gap with easy negatives.
        """
        siblings = [s for s in self.siblings_of.get(q_id, []) if s != q_id]
        negs = []

        if len(siblings) >= self.neg_per_sample:
            negs = random.sample(siblings, self.neg_per_sample)
        else:
            negs = siblings.copy()

        while len(negs) < self.neg_per_sample:
            for candidate in self.sample_easy_negatives(q_id):
                if candidate not in negs and candidate != q_id:
                    negs.append(candidate)
                    if len(negs) >= self.neg_per_sample:
                        break

        return negs[:self.neg_per_sample]

    def sample_hard_negatives(self, q_id):
        """
        Hard: Semantically similar but structurally dissimilar negative samples.
        Select samples from the semantic neighbor list that are neither themselves nor ancestors of the current sample. 
        If there are not enough samples, fall back to "easy".
        """
        avoid_struct = set(self.ancestors[q_id])
        avoid_struct.add(q_id)
        neighbors = self.sem_neighbors[q_id]
        negs = []

        for candidate in neighbors:
            if candidate in avoid_struct:
                continue
            negs.append(candidate)
            if len(negs) >= self.neg_per_sample:
                break

        if len(negs) < self.neg_per_sample:
            for candidate in self.sample_easy_negatives(q_id):
                if candidate not in negs and candidate != q_id:
                    negs.append(candidate)
                    if len(negs) >= self.neg_per_sample:
                        break

        return negs[:self.neg_per_sample]
