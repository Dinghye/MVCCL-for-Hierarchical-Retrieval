import nltk
import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    DEVICE,
    EMBED_DIM,
    FINETUNE_EPOCHS,
    LONG_DIST_THR,
    LR,
    MAX_ANCESTOR_DIST,
    MAX_DEPTH,
    PRETRAIN_EPOCHS,
    SEM_TOPK,
    TOP_K,
    set_global_seeds,
)
from datasets import LongDistanceDataset, RegularHrdataset
from embedding_utils import compute_semantic_neighbors, encode_texts_with_sbert
from evaluation import evaluate_by_distance, plot_recall_by_distance
from modeling import DualEncoder
from training import finetune_long_distance, train_mvccl, train_regular
from wordnet_utils import (
    assign_ids,
    build_parent_child_and_siblings,
    build_texts,
    build_wordnet_subtree,
    compute_ancestors_and_distances,
)

ROOT_LEMMA = "animal.n.01"


def prepare_wordnet_graph():
    print("Construct WordNet subtrees ...")
    graph, root = build_wordnet_subtree(root_lemma=ROOT_LEMMA, max_depth=MAX_DEPTH)
    syn2id, id2syn = assign_ids(graph)
    num_nodes = len(syn2id)
    print(f"Number of nodes in the subtree: {num_nodes}")

    print("Calculating ancestors and distances ...")
    ancestors, distances = compute_ancestors_and_distances(
        graph, syn2id, max_ancestor_dist=MAX_ANCESTOR_DIST
    )

    print("Build parent/child/siblings mapping ...")
    parents_of, children_of, siblings_of = build_parent_child_and_siblings(graph, syn2id)

    return {
        "graph": graph,
        "root": root,
        "syn2id": syn2id,
        "id2syn": id2syn,
        "ancestors": ancestors,
        "distances": distances,
        "parents_of": parents_of,
        "children_of": children_of,
        "siblings_of": siblings_of,
    }


def prepare_embeddings(id2syn):
    print("Constructing node text ...")
    texts = build_texts(id2syn)

    print("SBERT encoding text ...")
    base_embs = encode_texts_with_sbert(texts, model_name="all-MiniLM-L6-v2", device=DEVICE)
    print("base_embs:", base_embs.shape)

    print("Computing semantic neignbors...")
    sem_neighbors = compute_semantic_neighbors(base_embs, top_k=SEM_TOPK)
    return base_embs, sem_neighbors


def run_regular_stage(base_embs, all_q_ids, ancestors, distances):
    dataset = RegularHrdataset(all_q_ids, ancestors, distances)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = DualEncoder(base_embs.to(DEVICE), proj_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Regular training ...")
    train_regular(model, dataloader, optimizer, epochs=PRETRAIN_EPOCHS)
    print("\nRegular evaluation:")
    recall = evaluate_by_distance(model, all_q_ids, ancestors, distances, k=TOP_K)
    return model, recall


def run_hr_stage(base_embs, regular_state_dict, all_q_ids, ancestors, distances):
    model = DualEncoder(base_embs.to(DEVICE), proj_dim=EMBED_DIM).to(DEVICE)
    model.load_state_dict(regular_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.5)

    long_dataset = LongDistanceDataset(all_q_ids, ancestors, distances, long_dist_thr=LONG_DIST_THR)
    long_loader = DataLoader(
        long_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    print("Finetune(Only training long-distance samples)...")
    finetune_long_distance(model, long_loader, optimizer, epochs=FINETUNE_EPOCHS)

    print("\nHR Pretrain-Finetune evaluation:")
    recall = evaluate_by_distance(model, all_q_ids, ancestors, distances, k=TOP_K)
    return model, recall


def run_mvccl_stage(base_embs, all_q_ids, ancestors, distances, parents_of, siblings_of, sem_neighbors):
    model = DualEncoder(base_embs.to(DEVICE), proj_dim=EMBED_DIM).to(DEVICE)

    print("MVCCL training ...")
    train_mvccl(
        model,
        all_q_ids,
        ancestors,
        distances,
        parents_of,
        siblings_of,
        sem_neighbors,
        num_nodes=base_embs.size(0),
        base_lr=LR,
    )

    print("\nMVCCL evaluation:")
    recall = evaluate_by_distance(model, all_q_ids, ancestors, distances, k=TOP_K)
    return model, recall


def main():
    set_global_seeds()
    nltk.download("wordnet")

    wordnet_data = prepare_wordnet_graph()
    base_embs, sem_neighbors = prepare_embeddings(wordnet_data["id2syn"])
    all_q_ids = list(range(len(wordnet_data["syn2id"])))

    reg_model, reg_recall = run_regular_stage(
        base_embs, all_q_ids, wordnet_data["ancestors"], wordnet_data["distances"]
    )

    hr_model, hr_recall = run_hr_stage(
        base_embs,
        regular_state_dict=reg_model.state_dict(),
        all_q_ids=all_q_ids,
        ancestors=wordnet_data["ancestors"],
        distances=wordnet_data["distances"],
    )

    _, mvccl_recall = run_mvccl_stage(
        base_embs,
        all_q_ids,
        wordnet_data["ancestors"],
        wordnet_data["distances"],
        wordnet_data["parents_of"],
        wordnet_data["siblings_of"],
        sem_neighbors,
    )

    print("\n Paint Recall@Distance  ...")
    plot_recall_by_distance(
        result_dicts=[reg_recall, hr_recall, mvccl_recall],
        labels=["Regular", "HR Pretrain-Finetune", "MVCCL"],
        title="Hierarchical Retrieval: Recall@K vs Distance",
        save_path="recall_by_distance.png",
    )


if __name__ == "__main__":
    main()
