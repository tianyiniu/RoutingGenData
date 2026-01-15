
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from utils import SMALL_MODELS, LARGE_MODELS, read_json, write_json, l2_normalize


def _subject_groups(data: List[dict]) -> Dict[str, List[int]]:
    """Maps list of qdicts to dict[subject] --> list[qids]"""
    by_subject = defaultdict(list)
    for ex in data:
        by_subject[ex["category"]].append(ex["question_id"])
    return dict(by_subject)


def _load_subject_tables(subjects: List[str], load_dir: str):
    """
    Loads per-subject centroids, rankings, defaults, and cluster sizes.
    """
    subject_centroids, subject_rankings, subject_cluster_sizes = {}, {}, {}
    for s in subjects:
        if not os.path.isfile(f"{load_dir}/centroids_{s}.json"):
            continue

        centroids = read_json(f"{load_dir}/centroids_{s}.json")
        subject_centroids[s] = {cid: np.asarray(v, dtype=float) for cid, v in centroids.items()}
        subject_rankings[s] = read_json(f"{load_dir}/rankings_{s}.json")
        
        sizes_path = f"{load_dir}/cluster_sizes_{s}.json"
        if os.path.isfile(sizes_path):
            subject_cluster_sizes[s] = read_json(sizes_path)
        else:
            subject_cluster_sizes[s] = {}
            
    return subject_centroids, subject_rankings, subject_cluster_sizes


def _load_subject_representatives(load_dir: str) -> Dict[str, np.ndarray]:
    """
    Used for first-stage subject routing.
    """
    path = f"{load_dir}/subject_representatives.json"
    if not os.path.isfile(path):
        return {}
    
    data = read_json(path)
    return {subj: np.asarray(vec, dtype=float) for subj, vec in data.items()}


def _load_global_rankings(load_dir: str, models: List[str]) -> np.ndarray:
    """
    Returns array of scores in same order as models list.
    """
    path = f"{load_dir}/global_rankings.json"
    if not os.path.isfile(path):
        return np.ones(len(models)) / len(models)
    
    rankings = read_json(path)
    score_dict = dict(rankings)
    return np.array([score_dict.get(m, 0.0) for m in models], dtype=float)


def _route_to_subject(
        vec: np.ndarray,
        subject_representatives: Dict[str, np.ndarray]
    ) -> Tuple[Optional[str], Optional[float]]:
    """
    Returns:
        (subject_name, distance) or (None, None) if no representatives exist
    """
    if not subject_representatives:
        return None, None
    
    v = vec.astype(float)
    if v.ndim == 1:
        v = v[None, :]
    v = l2_normalize(v)[0]
    
    best_subj, best_dist = None, None
    
    for subj, rep_vec in subject_representatives.items():
        r = rep_vec.astype(float)
        if r.ndim == 1:
            r = r[None, :]
        r = l2_normalize(r)[0]
        dist = 1.0 - float(np.dot(v, r))
        
        if best_dist is None or dist < best_dist:
            best_subj, best_dist = subj, dist
    
    return best_subj, best_dist


def _nearest_centroid_any_subject(
        vec: np.ndarray,
        subject_centroids: Dict[str, Dict[str, np.ndarray]]
    ) -> Optional[Tuple[str, str, float]]:
    """
    Find the single nearest centroid across all subjects.
    Returns: (subject, cid, dist) or None if no centroids exist.
    """
    if not subject_centroids:
        return None

    v = vec.astype(float)
    if v.ndim == 1:
        v = v[None, :]
    v = l2_normalize(v)[0]

    best_subj, best_cid, best_dist = None, None, None

    for subj, centroids in subject_centroids.items():
        if not centroids:
            continue
        for cid, cvec in centroids.items():
            c = cvec.astype(float)
            if c.ndim == 1:
                c = c[None, :]
            c = l2_normalize(c)[0]
            dist = 1.0 - float(np.dot(v, c))
            if best_dist is None or dist < best_dist:
                best_subj, best_cid, best_dist = subj, cid, dist

    return best_subj, best_cid, best_dist


def _nearest_centroids(vec: np.ndarray, centroids: Dict[str, np.ndarray], topk: int) -> List[Tuple[str, float]]:
    """Find top-k nearest centroids within a subject."""
    if not centroids:
        return []

    v = vec.astype(float)
    if v.ndim == 1:
        v = v[None, :]
    v = l2_normalize(v)[0]

    dists = []
    for cid, cvec in centroids.items():
        c = cvec.astype(float)
        if c.ndim == 1:
            c = c[None, :]
        c = l2_normalize(c)[0]
        dist = 1.0 - float(np.dot(v, c))
        dists.append((cid, dist))

    dists.sort(key=lambda x: x[1])
    return dists[:min(topk, len(dists))]


def _cluster_sampling_scores(
        assigned_cluster_id: str,
        rankings_subject: Dict[str, List[Tuple[str, float]]],
        models_all: List[str],
    ) -> Optional[np.ndarray]:
    """Get raw model scores for a cluster."""
    rank_list = rankings_subject.get(assigned_cluster_id)
    if rank_list is None:
        return None

    per_model_acc = dict(rank_list)
    scores = np.array([float(per_model_acc.get(m, 0.0)) for m in models_all], dtype=float)
    return scores


def _cluster_sampling_scores_sharp(
        assigned_cluster_id: str,
        rankings_subject: Dict[str, List[Tuple[str, float]]],
        models_all: List[str],
        temperature: float = 0.1,
    ) -> Optional[np.ndarray]:
    """
    Lower temperature = sharper (more differentiated) scores.
    """
    rank_list = rankings_subject.get(assigned_cluster_id)
    if rank_list is None:
        return None

    per_model_acc = dict(rank_list)
    raw_scores = np.array([float(per_model_acc.get(m, 0.0)) for m in models_all], dtype=float)
    
    if raw_scores.sum() <= 0 or not np.all(np.isfinite(raw_scores)):
        return raw_scores
    
    # Apply softmax with temperature to sharpen differences
    shifted = raw_scores - raw_scores.max()  # Numerical stability
    exp_scores = np.exp(shifted / temperature)
    scores = exp_scores / exp_scores.sum()
    
    return scores


def _compute_distance_weights(dists: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    if len(dists) == 0:
        return np.array([])
    
    if len(dists) == 1:
        return np.array([1.0])
    
    # Convert distances to negative (so closer = higher)
    neg_dists = -dists / temperature
    
    # Softmax for numerical stability
    shifted = neg_dists - neg_dists.max()
    exp_dists = np.exp(shifted)
    weights = exp_dists / exp_dists.sum()
    
    return weights



def infer_router(
        test_data: List[dict],
        embeddings: Dict[int, List[float]],
        models: List[str],
        load_dir: str,
        save_path: str,
        use_two_stage_routing: bool = True,
        use_global_fallback: bool = True,
        distance_threshold: float = 0.35,
        use_sharp_scoring: bool = True,
        scoring_temperature: float = 0.1,
        distance_temperature: float = 0.1,
        k_nearest: int = 3,
        save_diagnostics: bool = True,
    ):

    by_subject = _subject_groups(test_data)
    subjects = list(by_subject.keys())
    subject_centroids, subject_rankings, subject_cluster_sizes = _load_subject_tables(subjects, load_dir)

    subject_representatives = _load_subject_representatives(load_dir)
    
    global_scores = _load_global_rankings(load_dir, models)
    
    q_to_subject = {ex["question_id"]: ex["category"] for ex in test_data}

    # Diagnostic tracking
    debug_stats = {
        'subject_correct': [],
        'nearest_dist': [],
        'consensus_spread': [],
        'top_model_score': [],
        'num_centroids_found': [],
        'qid': [],
        'true_subject': [],
        'predicted_subject': [],
        'used_fallback': [],
        'routing_method': [],
    }

    out = {}
    for ex in tqdm(test_data, desc="Inference"):
        qid = ex["question_id"]
        v = np.asarray(embeddings[qid], dtype=float)
        orig_subj = q_to_subject[qid]
        
        # Determine which subject to use for routing
        routing_method = None
        subj_for_cluster = None
        nearest_dist_global = None
        
        if use_two_stage_routing and subject_representatives:
            subj_for_cluster, subj_dist = _route_to_subject(v, subject_representatives)
            routing_method = 'two_stage'
            # Get distance to nearest centroid within subject
            if subj_for_cluster and subj_for_cluster in subject_centroids:
                centroids = subject_centroids[subj_for_cluster]
                nearest = _nearest_centroids(v, centroids, topk=1)
                if nearest:
                    nearest_dist_global = nearest[0][1]
        
        # Fallback: nearest centroid across all subjects
        else:
            subj_outs = _nearest_centroid_any_subject(v, subject_centroids)
            if subj_outs:
                subj_for_cluster, _, nearest_dist_global = subj_outs
            routing_method = 'nearest_any'

        # Track subject assignment
        subject_correct = 1 if subj_for_cluster == orig_subj else 0
        debug_stats['subject_correct'].append(subject_correct)
        debug_stats['qid'].append(qid)
        debug_stats['true_subject'].append(orig_subj)
        debug_stats['predicted_subject'].append(subj_for_cluster)
        debug_stats['nearest_dist'].append(nearest_dist_global)
        debug_stats['routing_method'].append(routing_method)

        # Handle case where no subject found
        if not subj_for_cluster or subj_for_cluster not in subject_centroids:
            models_and_scores = [(m, float(s), None) for m, s in zip(models, global_scores)]
            shuffle(models_and_scores)
            models_and_scores.sort(key=lambda x: -x[1])
            out[qid] = models_and_scores
            debug_stats['consensus_spread'].append(0.0)
            debug_stats['top_model_score'].append(float(global_scores.max()))
            debug_stats['num_centroids_found'].append(0)
            debug_stats['used_fallback'].append(1)
            continue

        centroids = subject_centroids[subj_for_cluster]
        rankings = subject_rankings[subj_for_cluster]
        nearest = _nearest_centroids(v, centroids, topk=k_nearest)

        debug_stats['num_centroids_found'].append(len(nearest))

        dists = np.array([d for _, d in nearest], dtype=float)
        w = _compute_distance_weights(dists, temperature=distance_temperature)

        # Compute weighted scores from clusters
        P = np.zeros(len(models), dtype=float)
        for (cid, _), wi in zip(nearest, w):
            if use_sharp_scoring:
                scores = _cluster_sampling_scores_sharp(
                    assigned_cluster_id=cid,
                    rankings_subject=rankings,
                    models_all=models,
                    temperature=scoring_temperature,
                )
            else:
                scores = _cluster_sampling_scores(
                    assigned_cluster_id=cid,
                    rankings_subject=rankings,
                    models_all=models,
                )
            
            if scores is None or not np.all(np.isfinite(scores)) or scores.sum() <= 0:
                continue
            P += wi * scores

        used_fallback = 0
        if use_global_fallback and nearest_dist_global is not None:
            if nearest_dist_global > distance_threshold:
                alpha = min((nearest_dist_global - distance_threshold) / 0.2, 1.0)
                
                if P.sum() > 0:
                    P_norm = P / P.sum()
                else:
                    P_norm = np.zeros_like(P)
                
                global_norm = global_scores / global_scores.sum() if global_scores.sum() > 0 else global_scores
                P = (1 - alpha) * P_norm + alpha * global_norm
                used_fallback = 1

        debug_stats['used_fallback'].append(used_fallback)

        # Track consensus spread
        if P.sum() > 0:
            debug_stats['consensus_spread'].append(float(P.max() - P.min()))
            debug_stats['top_model_score'].append(float(P.max()))
        else:
            debug_stats['consensus_spread'].append(0.0)
            debug_stats['top_model_score'].append(0.0)

        nearest_cid = nearest[0][0] if nearest else None
        models_and_scores = [(m, float(p), nearest_cid) for m, p in zip(models, P.tolist())]
        shuffle(models_and_scores)
        models_and_scores.sort(key=lambda x: -x[1], reverse=False)
        models_and_scores.sort(key=lambda x: x[1], reverse=True)

        out[qid] = models_and_scores

    
    # Save diagnostics
    if save_diagnostics:
        diag_path = save_path.replace('.json', '_diagnostics.json')
        # Convert numpy types to Python types for JSON serialization
        debug_stats_json = {
            k: [int(x) if isinstance(x, (np.integer, np.bool_)) else 
                float(x) if isinstance(x, np.floating) else x 
                for x in v]
            for k, v in debug_stats.items()
        }
        write_json(debug_stats_json, diag_path)
        print(f"Diagnostics saved to: {diag_path}")

    write_json(out, save_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-testt", "--test-task", type=str, required=True)
    parser.add_argument("-traint", "--train-task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, required=True)
    parser.add_argument("--models", type=str, required=True, choices=["small", "large"])
    
    parser.add_argument("--no-two-stage", action="store_true",
                        help="Disable two-stage routing")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable global fallback")
    parser.add_argument("--distance-threshold", type=float, default=0.5,
                        help="Distance threshold for fallback")
    parser.add_argument("--no-sharp-scoring", action="store_true",
                        help="Disable temperature-scaled scoring")
    parser.add_argument("--scoring-temp", type=float, default=0.1,
                        help="Temperature for score sharpening")
    parser.add_argument("--distance-temp", type=float, default=0.1,
                        help="Temperature for distance weighting")
    parser.add_argument("--k-nearest", type=int, default=3,
                        help="Number of nearest centroids")
    
    args = parser.parse_args()

    TRAIN_TASK = args.train_task
    TEST_TASK = args.test_task
    SPLIT = args.split
    EMBED_PATH = f"artifacts/{TEST_TASK}_only_question_embeddings_early.pt".replace("filtered", "large")

    models = LARGE_MODELS if args.models == "large" else SMALL_MODELS
    print(f"Using models: {models}")
    embeddings = torch.load(EMBED_PATH)
    test_data = read_json(f"Data/{TEST_TASK}/{SPLIT}.json")
    
    infer_router(
        test_data,
        embeddings,
        models,
        load_dir=f"./artifacts/{TRAIN_TASK}",
        save_path=f"./artifacts/{TEST_TASK}_agent_selections.json",
        use_two_stage_routing=not args.no_two_stage,
        use_global_fallback=not args.no_fallback,
        distance_threshold=args.distance_threshold,
        use_sharp_scoring=not args.no_sharp_scoring,
        scoring_temperature=args.scoring_temp,
        distance_temperature=args.distance_temp,
        k_nearest=args.k_nearest,
    )