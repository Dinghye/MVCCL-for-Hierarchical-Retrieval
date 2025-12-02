import torch
import torch.nn.functional as F


def contrastive_loss_inbatch(model, q_ids: torch.Tensor, pos_ids: torch.Tensor, temperature: float = 0.07):
    """
    InfoNCE/NT-Xent style in-batch contrast loss
    positive sample is (q_ids[i], pos_ids[i]), negative sample is batch other pos。
    """
    q = model.encode_query(q_ids)
    d = model.encode_doc(pos_ids)

    logits = q @ d.t() / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


def contrastive_loss_mvccl(
    model,
    q_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    neg_ids: torch.Tensor,
    temperature: float = 0.07,
    reduction='none'
):
    """
    InfoNCE with Explicit Negative Samples: Each sample has 1 positive sample + K negative samples.
    q_ids: [B], pos_ids: [B], neg_ids: [B, K]
    return per-sample loss (reduction='none') for weighting
    """
    q = model.encode_query(q_ids)
    pos = model.encode_doc(pos_ids)
    B = q.size(0)
    K = neg_ids.size(1)
    

    # positive sample
    sim_pos = torch.sum(q * pos, dim=-1) / temperature

    # neg sample
    neg_ids_flat = neg_ids.reshape(-1)
    neg_emb = model.encode_doc(neg_ids_flat)            # [B*K, D]
    neg_emb = neg_emb.view(B, K, -1)                    # [B, K, D]
    q_expand = q.unsqueeze(1)                           # [B, 1, D]
    sim_neg = torch.sum(q_expand * neg_emb, dim=-1) / temperature  # [B, K]

    logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)     # [B, 1+K]
    labels = torch.zeros(B, dtype=torch.long, device=q.device)     # 正样本在 index 0

    loss = F.cross_entropy(logits, labels, reduction=reduction)    # [B] 或 scalar
    return loss
    # flat_neg_ids = neg_ids.view(-1).to(q.device)
    # neg_emb = model.encode_doc(flat_neg_ids)
    # num_neg = flat_neg_ids.numel() // batch_size
    # neg_emb = neg_emb.view(batch_size, num_neg, -1)

    # q_expand = q.unsqueeze(1)
    # sim_neg = torch.sum(q_expand * neg_emb, dim=-1) / temperature

    # logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
    # labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
    # return F.cross_entropy(logits, labels)
