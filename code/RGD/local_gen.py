import os
import re
import gc
import torch
import random 
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from code.utils import read_json, write_json, MODEL_CACHE_DIR, AGENT_MAP
from code.RGD.subject_descriptions import MMLU_PRO, SUPER_GPQA, MEDMCQA, BBEH


MODEL_INSTRUCTION = r"""
You are an intelligent teaching assistant. Your current task is to generate questions for a multiple-choice exam. You will be given a description of the question category. Based on the description, generate one detailed **advanced graduate-level** question on similar topics.

**Constraints:**
1. Regardless of the number of choices in the input, ensure every generated question has exactly 4 choices (A, B, C, D).
2. Do NOT use JSON. Use the custom "Tagged Block" format defined below.

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

REMEMBER: FOLLOW THIS FORMAT EXACTLY. NO MARKDOWN CODE BLOCKS. ONCE AGAIN, ENSURE YOU INCLUDE ALL THE REQUIRED BLOCKS: [QUESTION], [OPTION A], [OPTION B], [OPTION C], [OPTION D], [ANSWER]!
"""

def parse_data(response: str) -> list[dict]:
    def extract_any(tags, text):
        # tags: list of possible tag names, e.g. ["OPTION A", "A"]
        tag_pattern = "|".join(re.escape(t) for t in tags)
        pattern = rf"\[(?:{tag_pattern})\](.*?)(?=\n\[[A-Z ]+\]|\Z)"
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
            q_text = extract_any(["QUESTION"], chunk)
            opt_a = extract_any(["OPTION A", "A"], chunk)
            opt_b = extract_any(["OPTION B", "B"], chunk)
            opt_c = extract_any(["OPTION C", "C"], chunk)
            opt_d = extract_any(["OPTION D", "D"], chunk)
            answer = extract_any(["ANSWER"], chunk)

            if all([q_text, opt_a, opt_b, opt_c, opt_d, answer]):
                questions.append({
                    "question": q_text,
                    "options": [opt_a, opt_b, opt_c, opt_d],
                    "answer": answer
                })
            else:
                print(f"Skipping incomplete chunk: {chunk}")

        except Exception as e:
            print(f"Error parsing chunk: {e}")
            continue    
    if len(questions) > 1:
        print(f"Extracted multiple questions sets: {len(questions)}")
    return questions


def select_question_ids(train_data: list[dict], requested_amount: int) -> list[str]:
    if requested_amount > len(train_data):
        print("Requested amount is out of bounds, sampling with replacement")

    data_for_selection = train_data

    subject_to_qids = {}
    for q in data_for_selection:
        qid = q["question_id"]
        subject = q["category"]
        subject_to_qids.setdefault(subject, []).append(qid)

    subjects = list(subject_to_qids.keys())
    for s in subjects:
        random.shuffle(subject_to_qids[s])

    selected_qids = []
    while len(selected_qids) < requested_amount:
        for subject in subjects[:]:
            if len(selected_qids) >= requested_amount:
                break

            if not subject_to_qids[subject]:
                subjects.remove(subject)
                continue

            qid = subject_to_qids[subject].pop()
            selected_qids.append(qid)

            if not subject_to_qids[subject] and len(selected_qids) < requested_amount:
                # Repopulate from original qids for that subject
                original_qids = [
                    q["question_id"] for q in train_data if q["category"] == subject
                ]
                random.shuffle(original_qids)
                subject_to_qids[subject].extend(original_qids)

    return selected_qids


def vllm_generate_responses_offline(model_fullname, llm, tokenizer, prompts, temperature, max_tokens):
    apply_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )

    if "qwen3" in model_fullname.lower():
        apply_kwargs["enable_thinking"] = False

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": MODEL_INSTRUCTION},
             {"role": "user", "content": p}],
            **apply_kwargs,
        ) for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=temperature, 
        top_p=0.95,
        max_tokens=max_tokens, 
    )
    outputs = llm.generate(formatted, sampling_params)
    return [output.outputs[0].text for output in outputs]


def run_inference(model_fullname, llm, tokenizer, temperature, max_tokens, prompt_dict, selected_data):
    prompts = []
    for d in selected_data:
        subj = d["category"]
        prompts.append(prompt_dict[subj])

    outputs = vllm_generate_responses_offline(
            model_fullname=model_fullname,
            llm=llm,
            tokenizer=tokenizer,
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens
        )
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_nickname", type=str, required=True)
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-d", "--cuda-devices", type=str, default="6,7")
    parser.add_argument("--num-seeds", type=int, default=30000)
    args = parser.parse_args()

    MODEL = args.model_nickname
    TASK = args.task
    SPLIT = args.split
    CUDA = args.cuda_devices

    NEW_DS_NAME = f"{TASK}_gen_{MODEL}_large"
    os.makedirs(f"./Data/{NEW_DS_NAME}", exist_ok=True)
    train_data = read_json(f"./Data/{TASK}/{SPLIT}.json")

    if TASK == "MMLU_Pro": 
        prompt_dict = MMLU_PRO
    elif TASK == "SuperGPQA": 
        prompt_dict = SUPER_GPQA
    elif TASK == "MedMCQA":
        prompt_dict = MEDMCQA
    elif TASK == "BBEH":
        prompt_dict = BBEH

    NUM_GPUS = len(CUDA.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA

    if MODEL in ["DeepSeekMath", "DpskMath", "QwenMath", "Y15"]: 
        max_tokens = 4096
    elif MODEL in ["Gemma", "BioLlama", "Qwen72", "OpenBio", "Llama31"]:
        max_tokens = 4096*2
    else: 
        max_tokens = 4096*4
        max_tokens = 4096

    model_fullname = AGENT_MAP[MODEL]
    llm = LLM(
        model = model_fullname,
        download_dir=MODEL_CACHE_DIR,
        tensor_parallel_size=NUM_GPUS,
        trust_remote_code=True,
        max_model_len = max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_fullname, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)   
    
    selected_qids = select_question_ids(train_data, args.num_seeds) 
    qid_to_row = {d["question_id"]: d for d in train_data}
    selected_data = [qid_to_row[qid] for qid in selected_qids]

    parsed_dataset = []
    current_idx = 0
    raw_generations = run_inference(
        model_fullname=model_fullname, 
        llm=llm, 
        tokenizer=tokenizer, 
        temperature=0.8, 
        max_tokens=max_tokens, 
        prompt_dict=prompt_dict, 
        selected_data=selected_data
    )
    
    print(f"Detected: {len(selected_data)} data and {len(raw_generations)} generations")
    for d, raw_output in zip(selected_data, raw_generations):
        try:
            parsed_outputs = parse_data(raw_output)  
            subject = d["category"]
            for q in parsed_outputs:
                q["question_id"] = current_idx
                q["category"] = subject 
                parsed_dataset.append(q)
                current_idx += 1
        except Exception: 
            continue
             
    # Write generated outputs
    new_question_file = f"./Data/{NEW_DS_NAME}/train.json"
    write_json(parsed_dataset, new_question_file)


    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Script completed. Exiting")
