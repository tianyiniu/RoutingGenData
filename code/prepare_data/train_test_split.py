import os
import argparse
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from code.utils import write_json, read_json, SMALL_MODELS, LARGE_MODELS

def split_dataset(data_filepath, dataset, split_ratio): 
    """Based on source dataset file, split and save q_dicts into train and test splits."""
    data = read_json(data_filepath)

    q_by_subject = defaultdict(list)
    for q_dict in tqdm(data): 
        q_by_subject[q_dict["category"]].append(q_dict)

    # Divide using split ratio
    train_qs, test_qs = [], []
    for subject, qs in q_by_subject.items():
        shuffle(qs)
        num_q = len(qs)
        split_idx = int(split_ratio * num_q)
        train, test = qs[:split_idx], qs[split_idx:]
        print(f"{subject}: total {num_q}, train: {len(train)}, test: {len(test)}")
        train_qs += train
        test_qs += test

    # Write to file 
    os.makedirs(f"./Data/{dataset}", exist_ok=True)
    write_json(train_qs, f"./Data/{dataset}/train.json")
    write_json(test_qs, f"./Data/{dataset}/test.json")


def split_dataset_via_qids(data_filepath, dataset, test_qids: list[int]): 
    """
        Based on source dataset file, split and save q_dicts into train and test splits.
        Save passed test_qids into test split, the rest into train split.
    """
    data = read_json(data_filepath)

    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise TypeError(
            f"{data_filepath} must be a list of dicts; got {type(data)} with first elem {type(data[0]) if isinstance(data, list) and data else None}"
        )

    # Divide using split ratio
    train_qs, test_qs = [], []
    for d in data:
        qid = d["question_id"]
        if qid in test_qids:
            test_qs.append(d)
        else: 
            train_qs.append(d)

    # Write to file 
    os.makedirs(f"./Data/{dataset}", exist_ok=True)
    write_json(train_qs, f"./Data/{dataset}/train.json")
    write_json(test_qs, f"./Data/{dataset}/test.json")



def split_model_responses(model_responses_dir, train_qs_path, test_qs_path, models=None):
    """Based on saved train and test data, split and save cached model responses."""
    train_qs = read_json(train_qs_path)
    print(len(train_qs))
    print(train_qs[0])
    test_qs = read_json(test_qs_path)

    train_ids = [q["question_id"] for q in train_qs]
    test_ids = [q["question_id"] for q in test_qs]
    print(f"Total questions: {len(train_ids)+len(test_ids)}")
    print(f"Found {len(train_ids)} train questions, and {len(test_ids)} test questions.")

    train_save_dir = f"{model_responses_dir}-train/"
    os.makedirs(train_save_dir, exist_ok=True)
    test_save_dir = f"{model_responses_dir}-test/"
    os.makedirs(test_save_dir, exist_ok=True)

    for filename in tqdm(os.listdir(model_responses_dir)):
        model_nickname = filename[:-5]

        if models and model_nickname not in models:
            continue

        model_responses = read_json(os.path.join(model_responses_dir, filename))

        model_train_qs, model_test_qs = [], []
        for q in model_responses: 
            q_id = q["question_id"]
            if q_id in train_ids:
                model_train_qs.append(q)
            elif q_id in test_ids: 
                model_test_qs.append(q)
            else: 
                continue # Question filtered out during keyword selection
                # print(f"UNKNOWN ID FOUND: {q_id}, model: {model_nickname}")
        write_json(model_train_qs, os.path.join(train_save_dir, filename))
        write_json(model_test_qs, os.path.join(test_save_dir, filename))
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset", type=str, choices=["MMLU_Pro", "SuperGPQA", "MedMCQA", "BBEH"])
    parser.add_argument("-m", "--models", type=str, choices=["large", "small"])
    parser.add_argument("-r", "--split-ratio", type=float, default=0.6)
    args = parser.parse_args()

    SPLIT_RATIO = args.split_ratio
    DATASET = args.dataset
    MODELS = args.models

    data_filepath = f"Data/{DATASET}/all.json"
    split_dataset(data_filepath, DATASET, SPLIT_RATIO)

    test_qids = read_json("./target_test.json")
    split_dataset_via_qids(data_filepath, DATASET, test_qids)

    model_responses_dir = f"./Model_Responses/{DATASET}-all"
    train_qs_path = f"Data/{DATASET}/train.json"
    test_qs_path = f"Data/{DATASET}/test.json"   
    split_model_responses(model_responses_dir, train_qs_path, test_qs_path, MODELS)