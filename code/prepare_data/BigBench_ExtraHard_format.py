import os
import re
import requests
from tqdm import tqdm
from code.utils import write_json

"""
Each q_dict object needs 
question_id: int
category: str
question: str
options: list[str]
answer: str (single letter choice)

Later scripts add fields "keywords" and "step_back"
"""

def split_mcq_options(raw_input: str):
    """
    Given a string containing multiple-choice options labeled (A)...(E),
    return a dict mapping each label ('A'..'E') to its corresponding text.
    """
    # Regex:
    #   \(([A-E])\)   -> match (A) ... (E), capture the letter
    #   \s*           -> optional whitespace after the label
    #   (.*?)         -> lazily capture everything (the option body)
    #   (?=\n\([A-E]\)\s|\Z) -> stop when we hit a newline + next label, or end of string
    pattern = re.compile(r"\(([A-Z])\)\s*(.*?)(?=\n\([A-Z]\)\s|\Z)", re.DOTALL)

    matches = pattern.findall(raw_input)
    return [text.strip() if "\n" not in text else text.split("\n")[0].strip() for label, text in matches]


def format_proved(ex, subset, current_global_index):
    new_q_dict = {}
    new_q_dict["question_id"] = current_global_index
    new_q_dict["category"] = subset
    new_q_dict["question"] = ex["input"]
    new_q_dict["options"] = ["proved", "disproved", "unknown"]

    gt_ans = ex["target"]
    if gt_ans == "proved":
        new_q_dict["answer"] = "A"
    elif gt_ans == "disproved":
        new_q_dict["answer"] = "B"
    elif gt_ans == "unknown":
        new_q_dict["answer"] = "C"
    else: 
        raise ValueError(f"Unknown option: {gt_ans}")
    return new_q_dict


def format_yes_no(ex, subset, current_global_index):
    new_q_dict = {}
    new_q_dict["question_id"] = current_global_index
    new_q_dict["category"] = subset
    new_q_dict["question"] = ex["input"]
    new_q_dict["options"] = ["Yes", "No", "Ambiguous"]

    gt_ans = ex["target"]
    if gt_ans == "Yes":
        new_q_dict["answer"] = "A"
    elif gt_ans == "No":
        new_q_dict["answer"] = "B"
    elif gt_ans == "Ambiguous":
        new_q_dict["answer"] = "C"
    else: 
        raise ValueError(f"Unknown option: {gt_ans}")
    return new_q_dict


def format_valid_invalid(ex, subset, current_global_index):
    new_q_dict = {}
    new_q_dict["question_id"] = current_global_index
    new_q_dict["category"] = subset
    new_q_dict["question"] = ex["input"].split("Options:")[0].strip("\n")
    new_q_dict["options"] = ["Valid", "Invalid"]
    new_q_dict["answer"] = "A" if ex["target"].lower() == "valid" else "B"
    return new_q_dict

def format_multiple_choice(ex, subset, current_global_index):
    new_q_dict = {}
    new_q_dict["question_id"] = current_global_index
    new_q_dict["category"] = subset
    new_q_dict["question"] = ex["input"].split("(A)")[0].strip("\n")
    new_q_dict["options"] = split_mcq_options(ex["input"])
    if len(ex["target"]) == 3:
        new_q_dict["answer"] = ex["target"][1]
    elif len(ex["target"]) == 1: 
        new_q_dict["answer"] = ex["target"]
    else: 
        raise ValueError(f"Answer improperly formatted: {ex["target"]}")
    return new_q_dict


SUBSETS_AND_FORMAT_FUNCS = {
    "bbeh_boardgame_qa": format_proved, # cite
    "bbeh_boolean_expressions": format_multiple_choice,
    "bbeh_causal_understanding": format_yes_no, #cite 1 2
    "bbeh_disambiguation_qa": format_multiple_choice,
    "bbeh_geometric_shapes": format_multiple_choice, # cite
    "bbeh_hyperbaton": format_multiple_choice,
    "bbeh_movie_recommendation": format_multiple_choice,
    "bbeh_shuffled_objects": format_multiple_choice,
    "bbeh_nycc": format_multiple_choice, #cite 1 2

    "bbeh_buggy_tables": None,
    "bbeh_dyck_languages": None,
    "bbeh_linguini": None,
    "bbeh_multistep_arithmetic": None,
    "bbeh_object_counting": None,
    "bbeh_object_properties": None,
    "bbeh_sarc_triples": None,
    "bbeh_spatial_reasoning": None,
    "bbeh_sportqa": None,
    "bbeh_temporal_sequence": None,
    "bbeh_time_arithmetic": None,
    "bbeh_web_of_lies": None,
    "bbeh_word_sorting": None,
    "bbeh_zebra_puzzles": None,
}

def fetch_json_from_url(subset: str):
    """
    Downloads a JSON file from a given URL, saves it to a temporary local file.
    """
    url = f"https://raw.githubusercontent.com/google-deepmind/bbeh/refs/heads/main/bbeh/benchmark_tasks/{subset}/task.json"
    print(url)
    try:
        response = requests.get(url, timeout=10) 
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"ERROR during download: {e}")
        return None


def main():
    global_index = 0
    processed_q_dicts = []

    for subset in tqdm(SUBSETS_AND_FORMAT_FUNCS):
        format_func = SUBSETS_AND_FORMAT_FUNCS[subset]
        if not format_func:
            continue

        split_data = fetch_json_from_url(subset)["examples"]
        for d in split_data:
            processed_q_dicts.append(format_func(d, subset, global_index))
            global_index += 1
    
    TASK_NAME = "BBEH"
    os.makedirs(f"./Data/{TASK_NAME}", exist_ok=True)
    write_json(processed_q_dicts, os.path.join(f"./Data/{TASK_NAME}", "all.json"))


if __name__ == "__main__":
    main()
