import os
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans as SKKMeans
from sklearn.metrics import silhouette_score
from utils import (
    LARGE_MODELS,
    SMALL_MODELS, 
    read_json, 
    write_json, 
    build_preds_dict, 
    get_majority_choice, 
    l2_normalize, 
    select_data
)


def _get_embedding_matrix(qids: List[int], emb: Dict[int, List[float]]) -> Tuple[np.ndarray, List[int]]:
    """Filter unwanted qids from embedding matrix, then stack into np array."""
    X = []
    for q in qids:
        vec = emb[q]
        X.append(vec)
    return np.asarray(X, dtype=float)


def _get_embedding_matrix_weighted(
        qids_and_weights: List[Tuple[int, float]], 
        emb: Dict[int, List[float]]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns: (X matrix, weights array, list of qids)
    """
    X = []
    weights = []
    valid_qids = []
    for qid, weight in qids_and_weights:
        if qid in emb:
            X.append(emb[qid])
            weights.append(weight)
            valid_qids.append(qid)
    return np.asarray(X, dtype=float), np.asarray(weights, dtype=float), valid_qids


def _kmeans_auto(
        X: np.ndarray, 
        seed: int = 0,
        min_k: int = 2, 
        max_k_cap: int = 5,
        silhouette_threshold: float = 0.05
    ):
    """
    Spherical k-means with automatic k for small datasets.
    Returns (centroids[L2-normalized], labels, chosen_k).
    """
    n_samples = X.shape[0]
    if n_samples <= 1:
        c = l2_normalize(X.mean(axis=0, keepdims=True))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    Xn = l2_normalize(X)

    n_unique = np.unique(np.round(Xn, decimals=12), axis=0).shape[0]
    if n_unique == 1:
        c = l2_normalize(Xn.mean(axis=0, keepdims=True))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    max_k = max(min(max_k_cap, n_samples - 1, n_unique - 1), 1)
    candidate_ks = [k for k in range(min_k, max_k + 1) if k >= 2]

    best = dict(score=-np.inf, k=None, km=None)
    for k in candidate_ks:
        try:
            km = SKKMeans(
                n_clusters=k, 
                n_init="auto",
                random_state=seed, 
                max_iter=200
            )
            labels = km.fit_predict(Xn)
            score = silhouette_score(Xn, labels, metric="cosine")
            if score > best["score"]:
                best.update(score=score, k=k, km=km)
        except Exception:
            continue

    if best["km"] is None or best["score"] < silhouette_threshold:
        c = l2_normalize(Xn.mean(axis=0, keepdims=True))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    C = l2_normalize(best["km"].cluster_centers_)
    return C, best["km"].labels_, best["k"]


def _weighted_kmeans_auto(
        X: np.ndarray,
        weights: np.ndarray,
        seed: int = 0,
        min_k: int = 2,
        max_k_cap: int = 5,
        silhouette_threshold: float = 0.05
    ):
    """
    Higher-weight samples have more influence on centroid positions.
    Returns (centroids[L2-normalized], labels, chosen_k).
    """
    n_samples = X.shape[0]
    if n_samples <= 1:
        c = l2_normalize(X.mean(axis=0, keepdims=True))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    Xn = l2_normalize(X)
    
    # Normalize weights to sum to n_samples (so they act as frequency weights)
    if weights.sum() > 0:
        weights_normalized = weights * (n_samples / weights.sum())
    else:
        weights_normalized = np.ones(n_samples)

    n_unique = np.unique(np.round(Xn, decimals=12), axis=0).shape[0]
    if n_unique == 1:
        c = l2_normalize(Xn.mean(axis=0, keepdims=True))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    max_k = max(min(max_k_cap, n_samples - 1, n_unique - 1), 1)
    candidate_ks = [k for k in range(min_k, max_k + 1) if k >= 2]

    best = dict(score=-np.inf, k=None, km=None, labels=None)
    for k in candidate_ks:
        try:
            km = SKKMeans(
                n_clusters=k,
                n_init="auto",
                random_state=seed,
                max_iter=200
            )
            labels = km.fit_predict(Xn, sample_weight=weights_normalized)
            score = silhouette_score(Xn, labels, metric="cosine")
            if score > best["score"]:
                best.update(score=score, k=k, km=km, labels=labels)
        except Exception:
            continue

    if best["km"] is None or best["score"] < silhouette_threshold:
        # Compute weighted mean for single cluster
        weighted_mean = np.average(Xn, axis=0, weights=weights_normalized)
        c = l2_normalize(weighted_mean.reshape(1, -1))
        labels = np.zeros(n_samples, dtype=int)
        return c, labels, 1

    # Recompute centroids using weights for better positioning
    C = np.zeros((best["k"], Xn.shape[1]))
    for cluster_idx in range(best["k"]):
        mask = best["labels"] == cluster_idx
        if mask.sum() > 0:
            cluster_weights = weights_normalized[mask]
            cluster_points = Xn[mask]
            C[cluster_idx] = np.average(cluster_points, axis=0, weights=cluster_weights)
    
    C = l2_normalize(C)
    return C, best["labels"], best["k"]


def _subject_groups(data: List[dict]) -> Dict[str, List[int]]:
    """Given a lst of q_dicts, create a dict that maps each subject to a list of qids."""
    by_subject = defaultdict(list)
    for ex in data:
        qid = ex["question_id"]
        subj = ex["category"]
        by_subject[subj].append(qid)
    return dict(by_subject)


def _majority_choice_correct(
        subject_qids: List[int],
        train_choices: Dict[str, Dict[str, int]],
        models: List[str]
    ) -> Dict[str, List[int]]:
    """
    Original: For each model, get questions that the model's selected choice was in the majority.
    """
    model_sets = {m: [] for m in models}
    qid_to_choice = {}

    for qid in subject_qids:
        choices = [train_choices[m][qid] for m in models]
        maj_choices = get_majority_choice(choices)
        qid_to_choice[qid] = maj_choices

    for qid in subject_qids:
        for m in models: 
            if train_choices[m][qid] in qid_to_choice[qid]:
                model_sets[m].append(qid)
    
    return model_sets


def _majority_choice_correct_weighted(
        subject_qids: List[int],
        train_choices: Dict[str, Dict[str, int]],
        train_consensus: Dict[str, Dict[str, float]],
        models: List[str],
        min_consensus_ratio: float = 0.5
    ) -> Dict[str, List[Tuple[int, float]]]:
    """
    Args:
        subject_qids: List of question IDs for this subject
        train_choices: model -> qid -> selected choice
        train_consensus: model -> qid -> consensus score
        models: List of model names
        min_consensus_ratio: Minimum agreement ratio to include question
        
    Returns:
        Dict mapping model -> list of (qid, weight) tuples
    """
    model_sets = {m: [] for m in models}
    
    for qid in subject_qids:
        choices = [train_choices[m][qid] for m in models]
        maj_choices = get_majority_choice(choices)
        
        # Compute agreement ratio as weight
        agreement_count = sum(1 for c in choices if c in maj_choices)
        agreement_ratio = agreement_count / len(models)
        
        # Skip low-consensus questions - they're noisy
        if agreement_ratio < min_consensus_ratio:
            continue
        
        # Use agreement ratio as weight
        weight = agreement_ratio
        
        for m in models:
            if train_choices[m][qid] in maj_choices:
                model_sets[m].append((qid, weight))
    
    return model_sets


def _assign_questions_to_centroids(
        subject_qids: List[int],
        centroids: Dict[str, np.ndarray],
        emb: Dict[int, List[float]]) -> Dict[str, List[int]]:
    """
    Assign every training question in subject to its nearest centroid using cosine distance.
    Returns: cluster_id -> list of qids
    """
    if not centroids:
        return {}
    cluster_to_qids = {cid: [] for cid in centroids}
    cid_list = list(centroids.keys())
    C = np.vstack([centroids[cid] for cid in cid_list])
    C = l2_normalize(C)

    for q in subject_qids:
        v = emb.get(q, None)
        if v is None:
            continue
        v = np.asarray(v, dtype=float)
        v = l2_normalize(v[None, :])[0]
        sims = C @ v
        j = int(np.argmax(sims))
        cluster_to_qids[cid_list[j]].append(q)
    return cluster_to_qids


def _compute_consensus_rankings_per_cluster(
        cluster_to_qids: Dict[str, List[int]],
        models: List[str],
        train_consensus: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
    """Compute model rankings for each cluster based on consensus scores."""
    rankings = {}
    for cid, qids in cluster_to_qids.items():
        if not qids:
            accs = [(m, 0.0) for m in models]
        else:
            accs = []
            denom = float(len(qids))
            for m in models:
                total_consensus_score = 0
                for q in qids:
                    total_consensus_score += train_consensus[m][q]
                mean_consensus_score = round(total_consensus_score/denom, 3)
                accs.append((m, mean_consensus_score))
        accs.sort(key=lambda x: (-x[1], x[0]))
        rankings[cid] = accs
    return rankings


def _merge_close_centroids(
        centroids: Dict[str, np.ndarray],
        seed_counts: Dict[str, int],
        merge_threshold: float = 0.15,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Reduces fragmentation and creates more robust clusters.
    
    Args:
        centroids: cluster_id -> centroid vector
        seed_counts: cluster_id -> number of points that formed this centroid
        merge_threshold: cosine distance below which to merge
        
    Returns:
        Merged centroids and seed counts
    """
    if len(centroids) <= 1:
        return centroids, seed_counts
    
    cids = list(centroids.keys())
    merged_into = {}
    
    # Compute pairwise distances
    for i, cid1 in enumerate(cids):
        if cid1 in merged_into:
            continue
            
        c1 = l2_normalize(centroids[cid1].reshape(1, -1))[0]
        
        for cid2 in cids[i+1:]:
            if cid2 in merged_into:
                continue
                
            c2 = l2_normalize(centroids[cid2].reshape(1, -1))[0]
            dist = 1.0 - np.dot(c1, c2)
            
            if dist < merge_threshold:
                # Merge cid2 into cid1 (keep the one with more seed points)
                if seed_counts.get(cid2, 0) > seed_counts.get(cid1, 0):
                    merged_into[cid1] = cid2
                    break
                else:
                    merged_into[cid2] = cid1
    
    # Build merged centroids
    merge_groups = defaultdict(list)  # target_cid -> list of cids to merge
    for cid in cids:
        target = cid
        while target in merged_into:
            target = merged_into[target]
        merge_groups[target].append(cid)
    
    new_centroids = {}
    new_seed_counts = {}
    
    for target_cid, group in merge_groups.items():
        if len(group) == 1:
            new_centroids[target_cid] = centroids[target_cid]
            new_seed_counts[target_cid] = seed_counts.get(target_cid, 0)
        else:
            # Weighted average of centroids by seed count
            vecs = [centroids[cid] for cid in group]
            weights = [seed_counts.get(cid, 1) for cid in group]
            total_weight = sum(weights)
            merged_vec = sum(v * w for v, w in zip(vecs, weights)) / total_weight
            merged_vec = l2_normalize(merged_vec.reshape(1, -1))[0]
            
            new_centroids[target_cid] = merged_vec
            new_seed_counts[target_cid] = sum(seed_counts.get(cid, 0) for cid in group)
    
    return new_centroids, new_seed_counts


def _recompute_centroids_from_assignments(
        cluster_to_qids: Dict[str, List[int]],
        embeddings: Dict[int, List[float]],
    ) -> Dict[str, np.ndarray]:
    """
    Recompute each centroid as the mean of its currently assigned question embeddings.
    Uses spherical mean: mean of L2-normalized embeddings, then L2-normalize the mean.
    """
    new_centroids = {}
    for cid, qids in cluster_to_qids.items():
        vecs = []
        for qid in qids:
            v = embeddings.get(qid)
            if v is not None:
                vecs.append(v)
        if not vecs:
            continue
        X = np.asarray(vecs, dtype=float)
        Xn = l2_normalize(X)
        c = l2_normalize(Xn.mean(axis=0, keepdims=True))[0]
        new_centroids[cid] = c
    return new_centroids


def _prune_redundant_centroids(
        centroids: Dict[str, np.ndarray],
        rankings: Dict[str, List[Tuple[str, float]]],
        cluster_to_qids: Dict[str, List[int]],
        similarity_threshold: float = 0.95
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, List], Dict[str, List[int]]]:
    """
    Only keeps centroids that provide differentiated routing signal.
    
    Args:
        centroids: cluster_id -> centroid vector
        rankings: cluster_id -> [(model, score), ...]
        cluster_to_qids: cluster_id -> list of assigned qids
        similarity_threshold: Jaccard similarity of top-3 above which to merge
        
    Returns:
        Pruned centroids, rankings, and cluster assignments
    """
    def top_k_similarity(r1: List[Tuple[str, float]], r2: List[Tuple[str, float]], k: int = 3) -> float:
        """Jaccard similarity of top-k models."""
        top1 = set(m for m, _ in r1[:k])
        top2 = set(m for m, _ in r2[:k])
        if not top1 or not top2:
            return 0.0
        return len(top1 & top2) / len(top1 | top2)
    
    cids = list(centroids.keys())
    keep = set(cids)
    
    for i, cid1 in enumerate(cids):
        if cid1 not in keep:
            continue
        for cid2 in cids[i+1:]:
            if cid2 not in keep:
                continue
            
            sim = top_k_similarity(rankings.get(cid1, []), rankings.get(cid2, []))
            
            if sim >= similarity_threshold:
                # Keep the one with more assigned questions
                n1 = len(cluster_to_qids.get(cid1, []))
                n2 = len(cluster_to_qids.get(cid2, []))
                if n2 > n1:
                    keep.discard(cid1)
                    # Reassign cid1's questions to cid2
                    cluster_to_qids[cid2] = cluster_to_qids.get(cid2, []) + cluster_to_qids.get(cid1, [])
                    break
                else:
                    keep.discard(cid2)
                    cluster_to_qids[cid1] = cluster_to_qids.get(cid1, []) + cluster_to_qids.get(cid2, [])
    
    return (
        {cid: centroids[cid] for cid in keep},
        {cid: rankings[cid] for cid in keep},
        {cid: cluster_to_qids[cid] for cid in keep}
    )


def _compute_global_rankings(
        all_qids: List[int],
        models: List[str],
        train_consensus: Dict[str, Dict[str, float]]
    ) -> List[Tuple[str, float]]:
    """
    Used as fallback when questions are far from all centroids.
    
    Returns:
        List of (model, mean_consensus_score) sorted by score descending
    """
    if not all_qids:
        return [(m, 0.0) for m in models]
    
    rankings = []
    for m in models:
        total = sum(train_consensus[m].get(qid, 0.0) for qid in all_qids)
        mean_score = total / len(all_qids)
        rankings.append((m, round(mean_score, 4)))
    
    rankings.sort(key=lambda x: (-x[1], x[0]))
    return rankings


def _compute_subject_representatives(
        subject_centroids: Dict[str, np.ndarray],
        subject_qids: List[int],
        embeddings: Dict[int, List[float]]
    ) -> np.ndarray:
    """
    Uses weighted average of cluster centroids, or mean of all question embeddings.
    
    Returns:
        L2-normalized representative vector for the subject
    """
    if subject_centroids:
        # Average of all centroids in this subject
        vecs = list(subject_centroids.values())
        rep = np.mean(vecs, axis=0)
    else:
        # Fallback: mean of all question embeddings
        vecs = [embeddings[qid] for qid in subject_qids if qid in embeddings]
        if vecs:
            rep = np.mean(vecs, axis=0)
        else:
            return None
    
    return l2_normalize(rep.reshape(1, -1))[0]


def train_router(
        train_data: List[dict],
        models: List[str],
        embeddings: Dict[int, List[float]],
        model_response_dir: str,
        save_dir: str,
        random_state: int = 0,
        use_weighted_clustering: bool = True,  
        merge_threshold: float = 0.15,         
        prune_similar_rankings: bool = True,   
        min_consensus_ratio: float = 0.5,   
    ):
    """
    Main training entry point.
    """
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=False)

    by_subject = _subject_groups(train_data)
    all_qids = [ex["question_id"] for ex in train_data]
    
    # Load consensus scores once for global rankings
    train_consensus = build_preds_dict(models, model_response_dir, field="consensus_score")
    train_choices = build_preds_dict(models, model_response_dir, field="selected_choice")
    
    global_rankings = _compute_global_rankings(all_qids, models, train_consensus)
    write_json(global_rankings, f"{save_dir}/global_rankings.json")
    print(f"Global rankings: {global_rankings}")  # Print top 3
    
    subject_representatives = {}
    
    num_single_clusters = 0
    num_merged = 0
    num_pruned = 0
    
    for subject, subj_qids in tqdm(by_subject.items(), desc="Training subjects"):

        if use_weighted_clustering:
            per_model_qs = _majority_choice_correct_weighted(
                subj_qids, train_choices, train_consensus, models,
                min_consensus_ratio=min_consensus_ratio
            )
        else:
            per_model_qs_unweighted = _majority_choice_correct(subj_qids, train_choices, models)
            # Convert to weighted format with uniform weights
            per_model_qs = {m: [(qid, 1.0) for qid in qids] 
                           for m, qids in per_model_qs_unweighted.items()}
        
        # Derive model-specific centroids and aggregate
        centroids = {}
        seed_counts = {}
        
        for m in models:
            qids_and_weights = per_model_qs[m]
            if len(qids_and_weights) == 0:
                continue
            
            if use_weighted_clustering:
                X, weights, valid_qids = _get_embedding_matrix_weighted(qids_and_weights, embeddings)
                if X.shape[0] == 0:
                    continue
                C, labels, best_k = _weighted_kmeans_auto(X, weights, seed=random_state)
            else:
                qids = [qid for qid, _ in qids_and_weights]
                X = _get_embedding_matrix(qids, embeddings)
                C, labels, best_k = _kmeans_auto(X, seed=random_state)
            
            if best_k == 1:
                num_single_clusters += 1

            counts = np.bincount(labels, minlength=C.shape[0])

            for j in range(C.shape[0]):
                cid = f"{subject}::model={m}::c{j}"
                centroids[cid] = C[j].astype(float)
                seed_counts[cid] = int(counts[j])

        if merge_threshold > 0:
            n_before = len(centroids)
            centroids, seed_counts = _merge_close_centroids(
                centroids, seed_counts, merge_threshold=merge_threshold
            )
            num_merged += n_before - len(centroids)

        cluster_to_qids = _assign_questions_to_centroids(subj_qids, centroids, embeddings)

        active_cids = {cid for cid, qids in cluster_to_qids.items() if qids}
        cluster_to_qids = {cid: cluster_to_qids[cid] for cid in active_cids}
        centroids = {cid: centroids[cid] for cid in active_cids}
        seed_counts = {cid: seed_counts.get(cid, 0) for cid in active_cids}

        rankings = _compute_consensus_rankings_per_cluster(cluster_to_qids, models, train_consensus)

        if prune_similar_rankings:
            n_before = len(centroids)
            centroids, rankings, cluster_to_qids = _prune_redundant_centroids(
                centroids, rankings, cluster_to_qids, similarity_threshold=0.95
            )
            num_pruned += n_before - len(centroids)

            rankings = _compute_consensus_rankings_per_cluster(cluster_to_qids, models, train_consensus)
            centroids = _recompute_centroids_from_assignments(cluster_to_qids, embeddings)

            for cid in list(seed_counts.keys()):
                if cid not in centroids:
                    seed_counts.pop(cid, None)
            for cid in centroids:
                seed_counts[cid] = seed_counts.get(cid, 0)  # or len(cluster_to_qids.get(cid, []))

        subject_rep = _compute_subject_representatives(centroids, subj_qids, embeddings)
        if subject_rep is not None:
            subject_representatives[subject] = subject_rep.tolist()

        # Save centroids
        centroids_json = {cid: centroids[cid].tolist() for cid in centroids}
        write_json(centroids_json, f"{save_dir}/centroids_{subject}.json")

        # Save rankings
        write_json(rankings, f"{save_dir}/rankings_{subject}.json")

        # Save cluster assignments
        cluster_assignments = {
            cid: [int(qid) for qid in cluster_to_qids[cid]]
            for cid in cluster_to_qids
        }
        write_json(cluster_assignments, f"{save_dir}/assigned_{subject}.json")

        # Save cluster sizes
        cluster_sizes = {
            cid: {
                "seed": int(seed_counts.get(cid, 0)),
                "assigned": int(len(cluster_to_qids.get(cid, []))),
            } for cid in centroids
        }
        write_json(cluster_sizes, f"{save_dir}/cluster_sizes_{subject}.json")

    write_json(subject_representatives, f"{save_dir}/subject_representatives.json")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, required=True)
    parser.add_argument("-nm", "--niche-method", type=str, default="majority_choice")
    parser.add_argument("-nq", "--num-questions", type=int, default=-1)
    parser.add_argument("--models", type=str, required=True, choices=["small", "large"])
    parser.add_argument("--no-weighted", action="store_true", help="Disable weighted clustering")
    parser.add_argument("--merge-threshold", type=float, default=0.15, help="Centroid merge threshold")
    parser.add_argument("--no-prune", action="store_true", help="Disable ranking-based pruning")
    parser.add_argument("--min-consensus", type=float, default=0.5, help="Minimum consensus ratio")
    args = parser.parse_args()

    TASK = args.task
    SPLIT = args.split
    EMBEDDINGS_PATH = f"artifacts/{TASK}_only_question_embeddings_early.pt".replace("filtered", "large")
    NUM_QUESTIONS = args.num_questions

    save_dir = f"./artifacts/{TASK}"
    models = LARGE_MODELS if args.models == "large" else SMALL_MODELS
    print(f"Using models: {models}")

    train_data = read_json(f"Data/{TASK}/{SPLIT}.json")
    if NUM_QUESTIONS > 0 and NUM_QUESTIONS <= len(train_data):
        train_data = select_data(train_data, NUM_QUESTIONS)
    print(f"## Number of training questions used: {len(train_data)}")

    embeddings = torch.load(EMBEDDINGS_PATH)
    model_response_dir = f"Model_Responses/{TASK}-{SPLIT}"

    train_router(
        train_data, models, embeddings,
        model_response_dir=model_response_dir,
        save_dir=save_dir, 
        random_state=0,
        use_weighted_clustering=not args.no_weighted,
        merge_threshold=args.merge_threshold,
        prune_similar_rankings=not args.no_prune,
        min_consensus_ratio=args.min_consensus,
    )