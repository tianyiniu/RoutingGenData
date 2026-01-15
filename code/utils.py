import re
import os
import json
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from math_verify import parse, verify
from transformers import AutoTokenizer
from collections import deque, defaultdict, Counter

load_dotenv()

MODEL_CACHE_DIR = ...
PROJECT_ROOT_DIR = ...
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


SMALL_MODELS = [
    "Gemma", "Qwen3-8", "Exaone", "GLM4", "DpskMath", "Yi15"
]

LARGE_MODELS = [
    "Llama70", "Qwen3-32", "GLM4-32", "Gemma3-27", "Exaone4-32", "OSS120"
]

AGENT_MAP = {
    # Large models
    "OSS120": "openai/gpt-oss-120b",
    "Qwen3-32": "Qwen/Qwen3-32B",
    "GLM4-32": "zai-org/GLM-4-32B-0414",
    "Llama70": "meta-llama/Llama-3.3-70B-Instruct",
    "Gemma3-27": "google/gemma-3-27b-it",
    "Exaone4-32": "LGAI-EXAONE/EXAONE-4.0-32B",

    # Small models
    "Gemma": "google/gemma-2-9b-it",
    "Exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    "GLM4": "zai-org/glm-4-9b-chat",
    "Qwen3-8": "Qwen/Qwen3-8B",
    "DpskMath": "deepseek-ai/deepseek-math-7b-instruct",
    "Yi15": "01-ai/Yi-1.5-9B-Chat-16K",
}


def build_preds_dict(models: list[str], responses_dir: str, field: str = "is_correct", run: int=0):
    """Assumes each file in responses_dir has the format <model name>.json"""
    preds = {}
    for m in models:
        try:
            model_responses = read_json(os.path.join(responses_dir, f"{m}.json"))
        except Exception as e:
            print(f"ERROR -- failed on {m}: {os.path.join(responses_dir, f"{m}.json")}")
            raise(e)

        preds[m] = {}
        for q in model_responses: 
            try:
                if field == "response":
                    preds[m][q["question_id"]] = q[f"{m}_response"]
                elif field == "selected_choice": 
                    preds[m][q["question_id"]] = q["selected_choice"].upper() if q["selected_choice"].isalpha() else random.choice(["A", "B", "C", "D"])
                else:
                    preds[m][q["question_id"]] = q[field]
            except Exception as e:
                print(f"ERROR -- failed in model {m} question {q}")
                raise(e)
    return preds


def build_correctness_dict(preds_dict):
    """Builds dictionary that maps question id to list of models that answered correctly"""
    models = list(preds_dict.keys())
    qids = list(preds_dict[models[0]].keys())
    output_dict = defaultdict(list)
    for m in models:
        for qid in qids: 
            if preds_dict[m][qid]:
                output_dict[qid].append(m)
    return output_dict


def l2_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if X.size == 0:
        return X
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n


def get_gpt_client(model_id):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client


def get_majority_choice(choices):
    counts = Counter(choices)
    if not counts:
        return []
    max_count = max(counts.values())
    return [i for i, count in counts.items() if count == max_count]


def select_data(train_data: list[dict], requested_amount: int) -> list[dict]:
    """
    Selects a specific amount of data from the training set, balancing 
    selection across categories via a round-robin approach.
    """

    if requested_amount > len(train_data):
        print(f"Requested amount ({requested_amount}) exceeds total data length ({len(train_data)}).")
        return train_data

    grouped_data = defaultdict(list)
    for item in train_data:
        grouped_data[item['category']].append(item)


    category_queue = deque()
    for category_items in grouped_data.values():
        random.shuffle(category_items)
        category_queue.append(category_items)


    selected_data = []
    while len(selected_data) < requested_amount:
        current_category_list = category_queue.popleft()
        item = current_category_list.pop()
        selected_data.append(item)
        if current_category_list:
            category_queue.append(current_category_list)
    return selected_data


def get_tokenizer(model_fullname: str, cache_dir: str = MODEL_CACHE_DIR):
    tokenizer = AutoTokenizer.from_pretrained(
        model_fullname,
        use_fast=True,              
        trust_remote_code=True,
        download_dir=cache_dir
    )
    return tokenizer


def trim_to_n_tokens(text: str, n: int, tokenizer):
    trimmed_ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=True)


def token_length(text: str, tokenizer):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def subject_model_accuracies(data, models, model_preds):
    """
    Compute per-subject accuracy for each model based on binary correctness in model_preds.
    Returns: subject -> (model -> accuracy in [0,1]).
    """
    per_sub_qids = defaultdict(list)
    for q in data:
        per_sub_qids[q["category"]].append(q["question_id"])

    acc = {}
    for s, qids in per_sub_qids.items():
        denom = max(1, len(qids))
        per_model = {}
        for m in models:
            num_correct = sum(model_preds[m].get(qid, 0) for qid in qids)
            per_model[m] = num_correct / denom
        acc[s] = per_model
    return acc


def mean_and_std(values, rounding=3):
    n = len(values)
    if n == 0:
        raise ValueError("Empty list")

    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n  # population variance
    std = var**0.5
    if not rounding:
        return mean, std
    return round(mean, rounding), round(std, rounding)


def softmax(input_lst, eps=1e-12):
    x = np.asarray(input_lst, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    x = np.clip(x, -700, 700)
    e = np.exp(x)
    p = e / e.sum()
    p = (p + eps) / (1.0 + eps * p.size)
    return p


def _normalize(s: str) -> str:
    if not isinstance(s, str):
        return s
    return (s.replace("\r\n", "\n")
             .replace("\r", "\n")
             .replace("\u2028", "\n")
             .replace("\u2029", "\n")
             .replace("\ufeff", "")
             .replace("\u200b", ""))

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, str):
        return _normalize(obj)
    return obj


def write_json(obj, file_name):
    obj = _sanitize(obj) 
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def get_number_choice(text):
    if not text:
        return "N/A"
    match = re.findall(r"answer is \((\d)\)", text)
    if match:
        return match[-1]
    else:
        match = re.findall(r"\((\d)\)", text)
        return match[-1] if match else "N/A"
    return "N/A"


def get_alphabet_choice(text: str, num_choice: int):
    if not text:
        return None

    choices = "".join(chr(65 + i) for i in range(num_choice))

    # Prefer "The answer is (X)" pattern
    match = re.search(rf"[Tt]he answer is\s*\(?([{choices}])\)?", text)
    if match:
        return match.group(1)

    # LaTeX boxed answer: \( \boxed{A} \)
    match = re.findall(rf"\\\(\s*\\boxed\{{([{choices}])\}}\s*\\\)", text)
    if match:
        return match[-1]

    # Fallback: last explicit "(X)" pattern
    match = re.findall(rf"\(([{choices}])\)", text)
    if match:
        return match[-1]

    # Fallback: isolated letter
    match = re.findall(rf"\b([{choices}])\b", text)
    return match[-1] if match else "N/A"


def get_true_false(text):
    if not text:
        return "N/A"
    match = re.findall(r"(true|false)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"


def get_yes_no(text):
    if not text:
        return "N/A"
    match = re.findall(r"(yes|no)", text, re.IGNORECASE)
    return match[-1].lower() if match else "N/A"


def get_keywords(output):
    if "Keywords" in output:
        keywords = output.split("Keywords:")[-1].split(",")
    else:
        keywords = output.split("keywords:")[-1].split(",")
    keywords = [i.strip().lower().replace(".", "") for i in keywords]
    return keywords


def is_math_equiv(ref, pred):
    # Test math equivalence of ref and pred, can also handle answer choices e.g., A vs. (A)
    try:
        if any(
            [
                verify(parse(f"${ref}$"), parse(f"${pred}$")),
                verify(parse(ref), parse(pred)),
                verify(parse(ref), parse(pred.replace("\\(", "").replace("\\)", ""))),
            ]
        ):
            return True
    except Exception:
        return False
    return False


def has_consensus(predictions):
    ref = predictions[0]
    for exp in predictions[1:]:
        if not ref == exp:
            return False
    return True


def last_boxed_only_string(string):
    if not string:
        return "N/A"
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = string
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return s


def parse_number(text):
    try:
        match = re.findall(r"\$?([0-9]+[\.,]?[0-9]*)", text)
        return float(match[-1].replace(",", "")) if match else "N/A"
    except Exception:
        print(text)


def parse_boxed(s):
    if not s:
        return "N/A"
    s = last_boxed_only_string(s)
    s = remove_boxed(s)
    s = parse_number(s)
    return s


def extract_first_int(text: str):
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None
