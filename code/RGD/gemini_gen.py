import os
import re
import json
import random 
import argparse
from tqdm import tqdm
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential
from code.utils import read_json, write_json
from code.RGD.subject_descriptions import MMLU_PRO, SUPER_GPQA, MEDMCQA, BBEH


MODEL_INSTRUCTION = r"""
You are an intelligent teaching assistant. Your current task is to generate questions for a multiple-choice exam. You will be given a description of the question category. Based on the description, generate 5 **advanced graduate-level** questions on similar topics.

**Constraints:**
1. Regardless of the number of choices in the input, ensure every generated question has exactly 4 choices (A, B, C, D).
2. Do NOT use JSON. Use the custom "Tagged Block" format defined below.
3. Separate each complete question with the delimiter '#####' (5 hashes).

**The Tagged Block Format:**
Use the following tags to structure your response. Content can span multiple lines.

[QUESTION]
<Write the question text here. Use LaTeX $...$ for math.>
[OPTION A]
<Text for Option A>
[OPTION B]
<Text for Option B>
[OPTION C]
<Text for Option C>
[OPTION D]
<Text for Option D>
[ANSWER]
<Single letter A, B, C, or D>

**Example Output:**
[QUESTION]
Calculate the limit of $f(x)$ as $x \to \infty$.
[OPTION A]
0
[OPTION B]
1
[OPTION C]
\infty
[OPTION D]
Undefined
[ANSWER]
B
#####
[QUESTION]
... (next question)

REMEMBER: FOLLOW THIS FORMAT EXACTLY. NO MARKDOWN CODE BLOCKS.
"""

def parse_data(response: str) -> list[dict]:
    def extract(tag, text):
        pattern = fr"\[{tag}\](.*?)(?=\[|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    

    questions = []

    if not response:
        print("Response empty")
        raise ValueError("Empty response received")

    raw_chunks = response.split('#####')

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        try:
            q_text = extract("QUESTION", chunk)
            opt_a = extract("OPTION A", chunk)
            opt_b = extract("OPTION B", chunk)
            opt_c = extract("OPTION C", chunk)
            opt_d = extract("OPTION D", chunk)
            answer = extract("ANSWER", chunk)

            # 3. Validation
            if all([q_text, opt_a, opt_b, opt_c, opt_d, answer]):
                questions.append({
                    "question": q_text,
                    "options": [opt_a, opt_b, opt_c, opt_d],
                    "answer": answer
                })
            else:
                # If a chunk is non-empty but fails validation, warn the user
                print(f"Skipping incomplete chunk: {chunk[:50]}...")

        except Exception as e:
            print(f"Error parsing chunk: {e}")
            continue
    
    return questions


def chunk_list(data, n):
    return [data[i:i + n] for i in range(0, len(data), n)]


def select_from_subjects(
    subject_to_qids: dict[str, list[int]],
    num_q_per_subject: int,
) -> list[int]:
    """Randomly select some number of questions from each subject, capped by population size."""

    def _safe_sample(population: list[int], k: int) -> list[int]:
        if not population or k <= 0:
            print("Returning empty list. Empty population or kegative k.")
            return []
        k = min(k, len(population))
        return random.sample(population, k)
    
    selected_qids: list[int] = []
    for subject, qids in subject_to_qids.items():
        selected_qids.extend(_safe_sample(qids, num_q_per_subject))
    return selected_qids


def select_question_ids(train_data: list[dict], requested_amount: int) -> list[str]:
    assert 1 <= requested_amount <= len(train_data), "Requested amount is out of bounds."

    subject_to_qids = {}
    for q in train_data:
        qid = q["question_id"]
        subject = q["category"]
        if subject not in subject_to_qids:
            subject_to_qids[subject] = []
        subject_to_qids[subject].append(qid)

    subjects = list(subject_to_qids.keys())
    for s in subjects:
        random.shuffle(subject_to_qids[s])
        
    selected_qids = []
    while len(selected_qids) < requested_amount:
        # Iterate over a copy of the subjects list so we can remove empty ones safely
        for subject in subjects[:]:
            if len(selected_qids) >= requested_amount:
                break
            
            qid = subject_to_qids[subject].pop()
            selected_qids.append(qid)
            
            if not subject_to_qids[subject]:
                subjects.remove(subject)
    return selected_qids


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def gemini_answer(
    model_name: str, 
    client: genai.Client,
    instruction: str,
    user_prompt: str,
) -> dict:

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=instruction
        )
    )
    usage = response.usage_metadata
    
    total_input_tokens = usage.prompt_token_count or 0
    cached_input_tokens = usage.cached_content_token_count or 0
    output_tokens = usage.candidates_token_count or 0
    standard_input_tokens = total_input_tokens - cached_input_tokens

    p_tks = standard_input_tokens
    c_tks = output_tokens

    RATE_INPUT_STANDARD = 0.30 / 1_000_000
    RATE_INPUT_CACHED   = 0.03 / 1_000_000
    RATE_OUTPUT         = 2.50 / 1_000_000

    cost_input = standard_input_tokens * RATE_INPUT_STANDARD
    cost_cached = cached_input_tokens * RATE_INPUT_CACHED
    cost_output = output_tokens * RATE_OUTPUT
    total_cost = cost_input + cost_cached + cost_output

    if not response.text:
        return None, 0, 0, 0

    return response.text, p_tks, c_tks, round(total_cost, 3)


def process_prompt(model_name, client, user_prompt, sys_prompt):
    response_text, p_tks, c_tks, cost = gemini_answer(
        model_name=model_name,
        client=client,
        instruction=sys_prompt,
        user_prompt=user_prompt
    )
    parsed_output = parse_data(response_text)
    return parsed_output, p_tks, c_tks, cost


def run_inference(model_name, client, prompt_dict, selected_data: list[dict]):
    prompts = []
    for d in selected_data:
        subj = d["category"]
        prompts.append(prompt_dict[subj])

    total_p_tks, total_c_tks, total_cost = 0, 0, 0
    results_by_index = [None] * len(prompts) 
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_index = {
            executor.submit(process_prompt, model_name, client, p, MODEL_INSTRUCTION): i 
            for i, p in enumerate(prompts)
        }

        for fut in tqdm(as_completed(future_to_index), total=len(prompts)):
            index = future_to_index[fut]
            try:
                resp, p_tks, c_tks, cost = fut.result()
                total_p_tks += p_tks
                total_c_tks += c_tks
                total_cost += cost
                results_by_index[index] = resp 
            except Exception as e:
                print(f"Worker failed for index {index}: {e}")
                results_by_index[index] = None

    print(f"Total prompt tokens: {total_p_tks}, total completion tokens: {total_c_tks}, cost: {total_cost} USDs")
    
    return results_by_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("--num-seeds", type=int, default=1000)
    parser.add_argument("--num-per-seed", type=int, default=5)
    args = parser.parse_args()


    CLIENT = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    TASK = args.task
    SPLIT = args.split
    
    NEW_DS_NAME = f"{TASK}_gen"
    os.makedirs(f"./Data/{NEW_DS_NAME}", exist_ok=True)

    train_data = read_json(f"./Data/{TASK}/{SPLIT}.json")

    ex_log_path = f"./Data/{NEW_DS_NAME}/execution_log.json"

    if TASK == "MMLU_Pro": 
        prompt_dict = MMLU_PRO
    elif TASK == "SuperGPQA": 
        prompt_dict = SUPER_GPQA
    elif TASK == "MedMCQA":
        prompt_dict = MEDMCQA
    elif TASK == "BBEH":
        prompt_dict = BBEH
    
    if os.path.isfile(ex_log_path):
        execution_log = read_json(ex_log_path)
    else: 
        selected_qids = select_question_ids(train_data, args.num_seeds) 
        chunked_qids = chunk_list(selected_qids, n=100)
        
        execution_log = {"current_idx": 0} 
        execution_log["num_chunks"] = len(chunked_qids)
        
        for idx, chunk in enumerate(chunked_qids):
            chunk_name = f"chunk{idx}"
            execution_log[chunk_name] = {
                "ids": chunk, "finished": False
            }
            
    new_question_file = f"./Data/{NEW_DS_NAME}/train.json"
    if os.path.isfile(new_question_file):
        parsed_dataset = read_json(new_question_file)
    else:
        parsed_dataset = []

    try:
        current_idx = execution_log["current_idx"]
    
        for chunk_idx in range(execution_log["num_chunks"]):
            chunk_name = f"chunk{chunk_idx}" 
            chunk_dict = execution_log[chunk_name]
            
            if chunk_dict["finished"]:
                continue
                
            chunk_qids = chunk_dict["ids"]
            print(f"Processing {chunk_name}...")

            selected_data = [d for d in train_data if d["question_id"] in chunk_qids]

            raw_generations = run_inference(
                model_name="gemini-2.5-flash",
                client=CLIENT, 
                prompt_dict=prompt_dict,
                selected_data=selected_data
            )

            for d, new_qs in zip(selected_data, raw_generations):
                if new_qs is None:
                    print(f"Skipping failed generation for QID: {d.get('question_id')}")
                    continue
                subject = d["category"] 
                
                for q in new_qs:
                    q["question_id"] = current_idx
                    q["category"] = subject
                    parsed_dataset.append(q)
                    current_idx += 1

            # Update log
            chunk_dict["finished"] = True
            execution_log["current_idx"] = current_idx
            
            write_json(parsed_dataset, new_question_file)
            write_json(execution_log, ex_log_path)
            
    except Exception as e: 
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:  
        write_json(parsed_dataset, new_question_file)
        write_json(execution_log, ex_log_path)
