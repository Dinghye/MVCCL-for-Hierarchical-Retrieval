from collections import defaultdict, deque

import networkx as nx
from nltk.corpus import wordnet as wn


def build_wordnet_subtree(root_lemma: str = "animal.n.01", max_depth: int = 7):
    """
    Build a WordNet subtree rooted at the given synset, 
    and return the directed graph (child -> parent) and the root node.
    """
    root = wn.synset(root_lemma)
    graph = nx.DiGraph()

    queue = deque([(root, 0)])
    visited = {root}

    while queue:
        node, depth = queue.popleft()
        if depth > max_depth:
            continue

        for child in node.hyponyms():
            if child not in visited:
                visited.add(child)
                queue.append((child, depth + 1))
            graph.add_edge(child, node)

    for syn in visited:
        if syn not in graph:
            graph.add_node(syn)

    return graph, root


def assign_ids(graph: nx.DiGraph):
    """Assign an integer ID to each synset in the subtree"""
    synsets = list(graph.nodes())
    id2syn = {i: syn for i, syn in enumerate(synsets)}
    syn2id = {syn: i for i, syn in id2syn.items()}
    return syn2id, id2syn


def compute_ancestors_and_distances(graph: nx.DiGraph, syn2id, max_ancestor_dist: int = 7):
    """
    Perform an upward Breadth-First Search (BFS) for each node,
    collecting the ancestors and distances.
    ancestors[q_id] = [anc_id1, anc_id2, ...](include itself, distance 0)
    distances[(q_id, anc_id)] = distance
    """
    ancestors = defaultdict(list)
    distances = {}

    for syn in graph.nodes():
        q_id = syn2id[syn]
        ancestors[q_id].append(q_id)
        distances[(q_id, q_id)] = 0

        queue = deque([(syn, 0)])
        visited = {syn}

        while queue:
            node, dist = queue.popleft()
            if dist >= max_ancestor_dist:
                continue

            for parent in graph.successors(node):
                if parent in visited:
                    continue
                visited.add(parent)
                new_dist = dist + 1
                anc_id = syn2id[parent]
                ancestors[q_id].append(anc_id)
                distances[(q_id, anc_id)] = new_dist
                queue.append((parent, new_dist))

    return ancestors, distances


def build_parent_child_and_siblings(graph: nx.DiGraph, syn2id):
    """
    Generate parent / child / siblings mapping, structural perspective for MVCCL.
    parents_of[node_id] = [parent_ids...]
    children_of[parent_id] = [child_ids...]
    siblings_of[node_id] = set(ids sharing at least one parent)
    """
    parents_of = defaultdict(list)
    children_of = defaultdict(list)
    for child, parent in graph.edges():
        c_id = syn2id[child]
        p_id = syn2id[parent]
        parents_of[c_id].append(p_id)
        children_of[p_id].append(c_id)

    siblings_of = defaultdict(set)
    for parent_id, child_ids in children_of.items():
        for child in child_ids:
            for sibling in child_ids:
                if sibling != child:
                    siblings_of[child].add(sibling)

    return parents_of, children_of, siblings_of


def synset_to_text(syn):
    """turn synset to simple text: lemma names + definition"""
    lemmas = " ".join(syn.lemma_names())
    definition = syn.definition()
    return f"{lemmas}. {definition}"


def build_texts(id2syn):
    """Build synset text descriptions"""
    return [synset_to_text(id2syn[i]) for i in range(len(id2syn))]



