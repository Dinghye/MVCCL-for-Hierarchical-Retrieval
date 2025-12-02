import torch
from torch.utils.data import DataLoader

from config import BATCH_SIZE, LR, MVC_STAGES, NEG_PER_SAMPLE
from datasets import MvCclDataset
from losses import contrastive_loss_inbatch, contrastive_loss_mvccl


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


def train_mvccl(
    model,
    all_q_ids,
    ancestors,
    distances,
    parents_of,
    siblings_of,
    sem_neighbors,
    num_nodes: int,
    base_lr: float = LR,
    batch_size: int = BATCH_SIZE,
    neg_per_sample: int = NEG_PER_SAMPLE,
    stages=MVC_STAGES,
):
    """
    3 Stage curriculum:
      Stage 1: easy negatives (random)
      Stage 2: medium negatives (siblings)
      Stage 3: hard negatives (semantic-similar but structurally far)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    for stage_name, stage_epochs in stages.items():
        print(f"\n[MVCCL] Stage = {stage_name}, epochs = {stage_epochs}")
        dataset = MvCclDataset(
            stage_name,
            all_q_ids,
            ancestors,
            distances,
            parents_of,
            siblings_of,
            sem_neighbors,
            num_nodes,
            neg_per_sample=neg_per_sample,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        model.train()
        for epoch in range(stage_epochs):
            total_loss = 0.0
            for batch in loader:
                q_ids, pos_ids, _, neg_ids = batch
                q_ids = q_ids.to(model.base_embeddings.device)
                pos_ids = pos_ids.to(model.base_embeddings.device)
                neg_ids = neg_ids.to(model.base_embeddings.device)

                loss = contrastive_loss_mvccl(model, q_ids, pos_ids, neg_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * q_ids.size(0)

            avg_loss = total_loss / len(loader.dataset)
            print(f"[MVCCL-{stage_name}] Epoch {epoch+1}/{stage_epochs} - loss={avg_loss:.4f}")
