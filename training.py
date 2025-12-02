import torch
import random
from torch.utils.data import DataLoader

# from config import BATCH_SIZE, LR, MVC_STAGES, NEG_PER_SAMPLE, RANK_MARGIN,DEVICE,TEMP_EASY,TEMP_MEDIUM,TEMP_HARD,LAMBDA_RANK,EASY_REPLAY_RATIO_HARD
from datasets import MvCclDataset
from losses import contrastive_loss_inbatch, contrastive_loss_mvccl
import torch.nn.functional as F


def train_regular(model, dataloader, optimizer, epochs: int):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            q_ids, pos_ids, _ = batch
            q_ids = q_ids.to(model.base_embeddings.device)
            pos_ids = pos_ids.to(model.base_embeddings.device)

            loss = contrastive_loss_inbatch(model, q_ids, pos_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * q_ids.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Regular] Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")


def finetune_long_distance(model, dataloader, optimizer, epochs: int):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            q_ids, pos_ids, _ = batch
            q_ids = q_ids.to(model.base_embeddings.device)
            pos_ids = pos_ids.to(model.base_embeddings.device)

            loss = contrastive_loss_inbatch(model, q_ids, pos_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * q_ids.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[Finetune-Long] Epoch {epoch+1}/{epochs} - loss={avg_loss:.4f}")


def sample_hier_pos(q_id, ancestors, distances,
                    near_max=1, mid_min=2, mid_max=3):
    """
    return (near_pos, mid_pos):
      near_pos: d <= near_max
      mid_pos:  mid_min <= d <= mid_max
    if cannot find pair, return None。
    """
    near = [x for x in ancestors[q_id] if distances[(q_id, x)] <= near_max]
    mid = [x for x in ancestors[q_id] if mid_min <= distances[(q_id, x)] <= mid_max]
    if len(near) == 0 or len(mid) == 0:
        return None
    return random.choice(near), random.choice(mid)


def train_mvccl_v3(
    model,
    all_q_ids,
    ancestors,
    distances,
    parents_of,
    siblings_of,
    sem_neighbors,
    num_nodes: int,
    base_lr: float,
    batch_size: int,
    neg_per_sample: int ,
    stages,
    easy_replay_ratio_hard : float,
    temp_easy: float ,
    temp_mid: float,
    temp_hard: float ,
    rank_margin: float ,
    lambda_rank: float ,
    device
):
    """
    MVCCL v3:
    - easy -> medium -> hard(+easy replay)
    - Negative sample filtering: excluding cases where dist <= 1
    - Positive sample weighting: weight = 3 for dist = 0, weight = 2 for dist = 1
    - Hard stage with slightly higher temperature + easy replay to prevent forgetting
    - Additional hierarchical ranking loss:
        Short-distance positive samples must be more similar than medium-distance positive samples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    for stage_name, stage_epochs in stages.items():
        print(f"\n[MVCCL v3] Stage = {stage_name}, epochs = {stage_epochs}")
        if stage_name == "hard":
            # hard stage: hard + easy replay
            hard_dataset = MvCclDataset(
                "hard",
                all_q_ids,
                ancestors,
                distances,
                parents_of,
                siblings_of,
                sem_neighbors,
                num_nodes,
                neg_per_sample=neg_per_sample
            )
            hard_loader = DataLoader(
                hard_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

            easy_dataset = MvCclDataset(
                "easy",
                all_q_ids,
                ancestors,
                distances,
                parents_of,
                siblings_of,
                sem_neighbors,
                num_nodes,
                neg_per_sample=neg_per_sample
            )
            easy_loader = DataLoader(
                easy_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            easy_iter = iter(easy_loader)

            for epoch in range(stage_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for batch in hard_loader:
                    # replay part of easy batch
                    if random.random() < easy_replay_ratio_hard:
                        try:
                            batch = next(easy_iter)
                        except StopIteration:
                            easy_iter = iter(easy_loader)
                            batch = next(easy_iter)
                        q_ids, pos_ids, dists, neg_ids = batch
                        temperature = temp_easy
                    else:
                        q_ids, pos_ids, dists, neg_ids = batch
                        temperature = temp_hard

                    q_ids = q_ids.to(device)
                    pos_ids = pos_ids.to(device)
                    neg_ids = neg_ids.to(device)

                    # constract learning（per-sample loss）
                    loss_vec = contrastive_loss_mvccl(
                        model, q_ids, pos_ids, neg_ids,
                        temperature=temperature,
                        reduction='none'
                    )  # [B]

                    # Structural conformality: Short-distance positive samples have greater weights
                    weights = []
                    for qi, pi in zip(q_ids.cpu().tolist(), pos_ids.cpu().tolist()):
                        dist = distances[(qi, pi)]
                        if dist == 0:
                            w = 3.0
                        elif dist == 1:
                            w = 2.0
                        else:
                            w = 1.0
                        weights.append(w)
                    weights = torch.tensor(weights, device=device)
                    loss_pos = (loss_vec * weights).mean()

                    # Hierarchical ranking loss: Close-range positive samples > Medium-range positive samples
                    rank_losses = []
                    for qi in q_ids.cpu().tolist():
                        pair = sample_hier_pos(qi, ancestors, distances,
                                               near_max=1, mid_min=2, mid_max=3)
                        if pair is None:
                            continue
                        near_id, mid_id = pair

                        q_emb = model.encode_query(torch.tensor([qi], device=device))
                        near_emb = model.encode_doc(torch.tensor([near_id], device=device))
                        mid_emb = model.encode_doc(torch.tensor([mid_id], device=device))

                        sim_near = torch.sum(q_emb * near_emb)
                        sim_mid = torch.sum(q_emb * mid_emb)
                        rloss = F.relu(rank_margin - (sim_near - sim_mid))
                        rank_losses.append(rloss)

                    if len(rank_losses) > 0:
                        loss_rank = torch.stack(rank_losses).mean()
                    else:
                        loss_rank = torch.tensor(0.0, device=device)

                    loss = loss_pos + lambda_rank * loss_rank

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * q_ids.size(0)
                    total_samples += q_ids.size(0)

                avg_loss = total_loss / max(total_samples, 1)
                print(f"[MVCCL-{stage_name}] Epoch {epoch+1}/{stage_epochs} - loss={avg_loss:.4f}")
        else:
            # easy / medium stage
            dataset = MvCclDataset(
                stage_name,
                all_q_ids,
                ancestors,
                distances,
                parents_of,
                siblings_of,
                sem_neighbors,
                num_nodes,
                neg_per_sample=neg_per_sample
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )
            temperature = temp_easy if stage_name == "easy" else temp_mid

            for epoch in range(stage_epochs):
                model.train()
                total_loss = 0.0
                total_samples = 0

                for batch in loader:
                    q_ids, pos_ids, dists, neg_ids = batch
                    q_ids = q_ids.to(device)
                    pos_ids = pos_ids.to(device)
                    neg_ids = neg_ids.to(device)

                    loss_vec = contrastive_loss_mvccl(
                        model, q_ids, pos_ids, neg_ids,
                        temperature=temperature,
                        reduction='none'
                    )

                    weights = []
                    for qi, pi in zip(q_ids.cpu().tolist(), pos_ids.cpu().tolist()):
                        dist = distances[(qi, pi)]
                        if dist == 0:
                            w = 3.0
                        elif dist == 1:
                            w = 2.0
                        else:
                            w = 1.0
                        weights.append(w)
                    weights = torch.tensor(weights, device=device)
                    loss_pos = (loss_vec * weights).mean()

                    # we can also add weak rank loss in easy/medium stage(optional)
                    rank_losses = []
                    for qi in q_ids.cpu().tolist():
                        pair = sample_hier_pos(qi, ancestors, distances,
                                               near_max=1, mid_min=2, mid_max=3)
                        if pair is None:
                            continue
                        near_id, mid_id = pair

                        q_emb = model.encode_query(torch.tensor([qi], device=device))
                        near_emb = model.encode_doc(torch.tensor([near_id], device=device))
                        mid_emb = model.encode_doc(torch.tensor([mid_id], device=device))

                        sim_near = torch.sum(q_emb * near_emb)
                        sim_mid = torch.sum(q_emb * mid_emb)
                        rloss = F.relu(rank_margin- (sim_near - sim_mid))
                        rank_losses.append(rloss)

                    if len(rank_losses) > 0:
                        loss_rank = torch.stack(rank_losses).mean()
                    else:
                        loss_rank = torch.tensor(0.0, device=device)

                    loss = loss_pos + lambda_rank * loss_rank

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * q_ids.size(0)
                    total_samples += q_ids.size(0)

                avg_loss = total_loss / max(total_samples, 1)
                print(f"[MVCCL-{stage_name}] Epoch {epoch+1}/{stage_epochs} - loss={avg_loss:.4f}")
