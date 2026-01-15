import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from utils import read_json, softmax, get_majority_choice, SMALL_MODELS, LARGE_MODELS, read_json, write_json

def power_normalize(a, gamma=1.0, eps=1e-8):
    a_safe = np.clip(a, eps, 1.0)
    w = a_safe ** gamma
    return w 


def min_max_normalize(element, min, max, eps=1e-6):
    return eps + (1-eps) * (element-min)/(max-min)


def compute_z_score(lp, mu, sig):
    """Return the normalized z-score for a single logprob value."""
    # Fallback if sigma is zero or not positive
    if not np.isfinite(sig) or sig <= 0:
        sig = 1.0
    return (lp - mu) / sig


def sample_models(models, scores, num_agents=3):
    assert len(models) == len(scores)
    scores = power_normalize(scores)
    weights = softmax(scores)
    return [str(s) for s in np.random.choice(models, p=weights, size=num_agents, replace=True)]


def rank_models(models, num_agents=3):
    if len(models) > num_agents:
        return models[:num_agents]
    else: 
        return models
    

def _find_min_max_cost(costs_dir):
    """
    Finds the minimum and maximum lengths in cost dir, useful for later min-max normalization.
    """
    nums = []
    for m, qid_to_len in costs_dir.items():
        nums += list(qid_to_len.values())
    return min(nums), max(nums)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, required=True)
    parser.add_argument("--num-training-questions", type=int, default=-1)
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--save-task", type=str, default="")
    parser.add_argument("--models", type=str, required=True, choices=["small", "large"])
    args = parser.parse_args()

    TASK = args.task
    SPLIT = args.split
    MODEL_RESPONSE_DIR = f"Model_Responses/{TASK}-{SPLIT}"
    CACHE_PATH = f"artifacts/{TASK}_agent_selections.json"
    LOAD_DIR = f"artifacts/{TASK}"

    top_n = 3
    dataset_filepath = f"./Data/{TASK}/{SPLIT}.json"
    models = LARGE_MODELS if args.models == "large" else SMALL_MODELS
    print(f"Using models: {models}")
    q_dicts = read_json(dataset_filepath)
    cached_stats = read_json(f"./artifacts/{TASK}-train_stats.json".replace("filtered", "large"))

    qids = [q["question_id"] for q in q_dicts]
    n_q = len(qids)

    # Processed cached model rankings
    raw_rankings = read_json(CACHE_PATH)
    cached_rankings = {}
    for qid, model_tups in raw_rankings.items(): 
        qid = int(qid)
        model_tups = [(m, float(score)) for m, score, nearest_cid in model_tups]
        cached_rankings[qid] = model_tups


    answer_dict = {q["question_id"]: q["answer"] for q in q_dicts}
    pred_dict, choice_dict, lp_dict = {}, {}, {}
    for model in models:
        model_response_logs = read_json(os.path.join(MODEL_RESPONSE_DIR, f"{model}.json"))
        pred_dict[model] = {q["question_id"]: q["is_correct"] for q in model_response_logs}
        choice_dict[model] = {q["question_id"]: q["selected_choice"] for q in model_response_logs}
        lp_dict[model] = {q["question_id"]: q["avg_logprob"] for q in model_response_logs}


    # Trackers for selection accuracies
    selection_logs_ranked = []
    selection_logs_sampled = []
    top1_hits = 0
    rank_top3_hits = 0
    sample_top3_hits = 0
    ranked_maj_vote_hits = 0
    sampled_maj_vote_hits = 0

    for q_dict in tqdm(q_dicts):
        qid = q_dict["question_id"]
        model_tups = cached_rankings[qid]
        models_ranked = [tup[0] for tup in model_tups]
        scores = [tup[1] for tup in model_tups]

        # Top model selection accuracy
        top_model = models_ranked[0]
        top1_hits += 1 if pred_dict[top_model][qid] else 0
        
    
        # Ranking selection baselines
        ranked_selection = rank_models(models_ranked, top_n)
        sample_selection = sample_models(models_ranked, scores, num_agents=top_n)
        selection_logs_ranked.append(ranked_selection)
        selection_logs_sampled.append(sample_selection)

        qid = q_dict["question_id"]
        rank_top3_hits += 1 if any(pred_dict[m][qid] for m in ranked_selection) else 0
        sample_top3_hits += 1 if any(pred_dict[m][qid] for m in sample_selection) else 0

        try:
            # Consensus voting
            rank_choices = [choice_dict[m][qid] for m in ranked_selection]
            if (len(set(rank_choices))) == 3:
                # Get z_score, normalized using cached mu and sigma for model
                m_and_z = [] # [(model, z_score), ...]
                for m in ranked_selection:
                    mu, sig = cached_stats[m]
                    lp = lp_dict[m][qid]
                    z_score = compute_z_score(lp, mu, sig)
                    m_and_z.append((m, z_score))
                    m_and_z = sorted(m_and_z, key=lambda x: x[1], reverse=True)
                    best_m = m_and_z[0][0]

                ranked_maj_vote_hits += 1 if q_dict["answer"] == choice_dict[best_m][qid] else 0

            else:
                ranked_maj_vote_hits += 1 if q_dict["answer"] == get_majority_choice(rank_choices)[0] else 0

            sample_choices = [choice_dict[m][qid] for m in sample_selection]
            sampled_maj_vote_hits += 1 if q_dict["answer"] == get_majority_choice(sample_choices)[0] else 0

        except KeyError as e:
            print("KeyError found", e)
            rank_choices = [pred_dict[m][qid] for m in ranked_selection]
            ranked_maj_vote_hits += 1 if 1 == get_majority_choice(rank_choices)[0] else 0

            sample_choices = [pred_dict[m][qid] for m in sample_selection]
            sampled_maj_vote_hits += 1 if 1 == get_majority_choice(sample_choices)[0] else 0


    top1_acc = top1_hits / n_q

    rank_top3_acc = rank_top3_hits / n_q
    rank_maj_vote_acc = ranked_maj_vote_hits / n_q

    sample_top3_acc = sample_top3_hits / n_q
    sample_maj_vote_acc = sampled_maj_vote_hits / n_q


    print("=== Selection Method Accuracies ===")
    top1_acc = 100*round(top1_acc, 3)
    rank_maj_vote_acc = 100*round(rank_maj_vote_acc, 3)
    print(f"Rank top 1 acc: {top1_acc}")
    print(f"Rank: Majority vote: {rank_maj_vote_acc}")

    if args.num_training_questions and args.save_task and args.method:

        final_acc = round(rank_maj_vote_acc, 5)
        method = args.method
        stats_file = f"./{args.save_task}_{method}_top3_scaling_stats.json"
        if os.path.isfile(stats_file):
            existing_stats = read_json(stats_file)
        else:
            existing_stats = {}
        num_training_questions = str(args.num_training_questions)
        if num_training_questions in existing_stats:
            existing_stats[num_training_questions].append(final_acc)
        else: 
            existing_stats[num_training_questions] = [final_acc]
        write_json(existing_stats, stats_file)


        final_acc = round(top1_acc, 5)
        method = args.method
        stats_file = f"./{args.save_task}_{method}_top1_scaling_stats.json"
        if os.path.isfile(stats_file):
            existing_stats = read_json(stats_file)
        else:
            existing_stats = {}
        num_training_questions = str(args.num_training_questions)
        if num_training_questions in existing_stats:
            existing_stats[num_training_questions].append(final_acc)
        else: 
            existing_stats[num_training_questions] = [final_acc]
        write_json(existing_stats, stats_file)