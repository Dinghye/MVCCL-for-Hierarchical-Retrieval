from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import torch


@torch.no_grad()
def evaluate_by_distance(model, all_q_ids, ancestors, distances, k: int):
    model.eval()
    num_nodes = model.base_embeddings.size(0)

    all_doc_ids = torch.arange(num_nodes, device=model.base_embeddings.device)
    all_doc_emb = model.encode_doc(all_doc_ids)

    stat_hit = defaultdict(int)
    stat_total = defaultdict(int)

    for q_id in all_q_ids:
        q_id_tensor = torch.tensor([q_id], device=model.base_embeddings.device)
        q_emb = model.encode_query(q_id_tensor)
        scores = (q_emb @ all_doc_emb.t()).squeeze(0)

        _, topk_indices = torch.topk(scores, k=k, largest=True)
        topk_indices = topk_indices.cpu().numpy().tolist()

        for anc_id in ancestors[q_id]:
            dist = distances[(q_id, anc_id)]
            stat_total[dist] += 1
            if anc_id in topk_indices:
                stat_hit[dist] += 1

    recall_by_dist = {}
    print(f"=== Recall@{k} by structural distance ===")
    for dist in sorted(stat_total.keys()):
        total = stat_total[dist]
        hit = stat_hit[dist]
        recall = hit / total if total > 0 else 0.0
        recall_by_dist[dist] = recall
        print(f"  distance={dist}: recall={recall*100:.2f}%  (hit={hit}, total={total})")

    total_all = sum(stat_total.values())
    hit_all = sum(stat_hit.values())
    overall = hit_all / total_all if total_all > 0 else 0.0
    print(f"  OVERALL: recall={overall*100:.2f}% (hit={hit_all}, total={total_all})")

    return recall_by_dist


def plot_recall_by_distance(result_dicts, labels, title: str, save_path: Optional[str] = None):
    """
    result_dicts: [recall_dict1, recall_dict2, ...], like {distance: recall_value}
    labels: ["Regular", "HR-PF", "MVCCL"]
    """
    plt.figure(figsize=(8, 5))

    for recall_dict, label in zip(result_dicts, labels):
        distances = sorted(recall_dict.keys())
        recalls = [recall_dict[d] for d in distances]
        plt.plot(distances, recalls, marker="o", linewidth=2, label=label)

    plt.xlabel("Structural Distance d_S", fontsize=12)
    plt.ylabel("Recall@K", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()
