import os
import json
import math
import argparse
import numpy as np
from random import choice
from code.utils import LARGE_MODELS, SMALL_MODELS, ALPHABET, read_json, write_json

# Inputs:
# pred_dict[model][qid] = predicted option label (e.g., 'A','B','C','D')
# logprob_dict[model][qid] = float logprob confidence for that prediction
# classes = list of all option labels, e.g., ['A','B','C','D']

def robust_location_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))  # median absolute deviation
    # Consistent estimator for std under normality
    sigma = 1.4826 * mad
    if sigma == 0 or not np.isfinite(sigma):
        sigma = np.nanstd(x) or 1.0
    return med, sigma or 1.0


def zscore_per_model(logprob_dict):
    # Returns z[model][qid]
    # I.e., z_dict[model] -> q_dict
    # q_dict[question id] -> z score
    z = {}
    m_to_stats = {}
    for m, d in logprob_dict.items():
        vals = np.array(list(d.values()), dtype=float)  # lp of all questions
        mu, sig = robust_location_scale(vals)
        z[m] = {
            q: (lp - mu) / (sig if sig > 0 else 1.0) for q, lp in d.items()
        }  # normalized lp
        m_to_stats[m] = (mu, sig)
    return z, m_to_stats


def z_to_weight(z, scale=1.0, clamp=(0.01, 0.99)):
    # Map z to [0,1] with logistic; scale controls steepness.
    w = 1.0 / (1.0 + math.exp(-scale * z))
    lo, hi = clamp
    return min(max(w, lo), hi)


def per_question_scores_consensus(
    pred_dict, logprob_dict, classes, z_scale=1.0, tau=1.0
):
    # z-normalize per model
    z, _ = zscore_per_model(logprob_dict)

    qids = sorted(set().union(*[set(d.keys()) for d in pred_dict.values()]))  # All ids
    scores_dict = {m: {} for m in pred_dict}  # scores_dict[model][q_id] = score

    for q in qids:
        # aggregate votes
        raw = {c: 0.0 for c in classes}
        for m, preds in pred_dict.items():  # model, preds[q_id] = class
            if q in preds and q in z[m]:
                w = z_to_weight(z[m][q], scale=z_scale)
                raw[preds[q]] += w

        # posterior over classes (softmax of raw weights)
        vals = np.array([raw[c] for c in classes], dtype=float)
        mval = np.max(vals)
        exps = np.exp((vals - mval) / max(tau, 1e-6))
        p = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(classes)
        post = {c: p[i] for i, c in enumerate(classes)}

        # assign each model its posterior on its prediction
        for m, preds in pred_dict.items():
            if q in preds:
                c = preds[q]
                scores_dict[m][q] = float(post.get(c, 0.0))
    return scores_dict


def main(task, split, models, num_choices, classes):
    print("Building model logprob and pred dictionary")
    # model_responses_dict[model][qid] = 1 or 0
    logprob_dict = {model: None for model in models}
    pred_dict = {model: None for model in models}
    for model in models:
        model_responses_path = f"Model_Responses/{task}-{split}/{model}.json"
        with open(model_responses_path, "r", encoding="utf-8") as f:
            model_response_logs = json.load(f)

        pred_dict[model] = {
            q["question_id"]: q["selected_choice"] if q["selected_choice"] in classes else choice(classes) for q in model_response_logs
        }
        logprob_dict[model] = {
            q["question_id"]: q["avg_logprob"] for q in model_response_logs
        }

    scores_dict = per_question_scores_consensus(pred_dict, logprob_dict, classes)
    z, m_to_stats = zscore_per_model(logprob_dict)

    if split == "train":
        STATS_SAVE_PATH = f"./artifacts/{task}-{split}_stats.json"
        
        # Check if the file already exists
        if os.path.exists(STATS_SAVE_PATH):
            # Load the existing data
            existing_stats = read_json(STATS_SAVE_PATH)
            # Update existing data with new stats (new keys added, existing keys overwritten)
            existing_stats.update(m_to_stats)
            # Point m_to_stats to the combined dictionary
            m_to_stats = existing_stats
        
        # Write the updated dictionary back to the file
        write_json(m_to_stats, STATS_SAVE_PATH)
    

    # Load calculated scores into model results .json file
    for model in models:
        model_log_path = f"Model_Responses/{task}-{split}/{model}.json"

        model_logs = read_json(model_log_path)
        for log in model_logs:
            log_id = log["question_id"]
            log["consensus_score"] = scores_dict[model][log_id]
            log["z_score"] = z[model][log_id]
        write_json(model_logs, model_log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, )
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-m", "--models", type=str, choices=["large", "small"])
    args = parser.parse_args()

    TASK = args.task
    SPLIT = args.split
    MODELS = args.models

    models = LARGE_MODELS if MODELS == "large" else SMALL_MODELS
    num_choices = 20
    classes = [let for let in ALPHABET[:num_choices]]

    t_and_s = [(TASK, SPLIT)]
    for task, split in t_and_s:
        print(f"Current working on {task}:{split}")
        main(task, split, models, num_choices, classes)
