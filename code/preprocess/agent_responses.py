"""
Evaluate an individual agent on dataset. Saves the model's response, top-token avg. token logprob for each question. Also creates the oracle skill profile for the model.
"""

import os
import gc
import torch
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from code.utils import ALPHABET, MODEL_CACHE_DIR, AGENT_MAP, MODELS_OF_INTEREST, SMALL_MODELS, read_json, write_json, get_alphabet_choice, is_math_equiv, trim_to_n_tokens


def create_prompt(q_dict): 
    # Compose list of choices
    question = q_dict["question"]
    choices = q_dict.get("options", None)

    if choices:
        choice_str = "\nChoices:\n"
        for letter, choice in zip(ALPHABET, choices): 
            choice_str += f"({letter}) {choice}\n"
    else:
        choice_str = ""
    final_prompt = question + choice_str + "\nThink through the problem and provide your step-by-step reasoning. After that, if the question is a multiple choice problem, print 'The answer is (X)', where X is the answer choice (one capital letter), at the end of your response. If the question is a calculation question that a numerical output, print 'The answer is (X.X)', where X.X is a float representing the result of the calculation. If the question requires you do output a list of objects, print 'The answer is (X Y Z ...)' where X, Y, and Z represents and object in the list, delimited by a single space."
    return final_prompt


def vllm_generate_responses_offline(model_fullname, llm, tokenizer, prompts, temperature, max_tokens):
    apply_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    # Heuristic: only Qwen3 tokenizers understand enable_thinking
    # Adjust this condition to whatever your AGENT_MAP actually uses.
    if "qwen3" in model_fullname.lower():
        apply_kwargs["enable_thinking"] = False
    elif "oss" in model_fullname.lower():
        apply_kwargs["reasoning_effort"] = "low"

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": trim_to_n_tokens(p, max_tokens-100, tokenizer)}],
            **apply_kwargs,
        )
        for p in prompts
    ]

    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens, 
        logprobs=1
    )
    outputs = llm.generate(formatted, sampling_params)

    results = []
    for output in outputs:
        text = output.outputs[0].text
        token_logprobs = []
        for token in output.outputs[0].logprobs:
            _, top_lp = list(token.items())[0]
            token_logprobs.append(top_lp.logprob)
        results.append({"text": text, "avg_logprobs": np.mean(token_logprobs)})
    return results, 0, 0


def process_questions(q_dicts, model_nickname, task, llm, tokenizer, max_tokens):

    model_fullname = AGENT_MAP[model_nickname]
    prompts = [create_prompt(q) for q in q_dicts]

    responses, _, _ = vllm_generate_responses_offline(
        model_fullname, 
        llm, 
        tokenizer, 
        prompts, 
        temperature=0.7,
        max_tokens=max_tokens
    )

    response_texts = [r["text"] for r in responses]
    avg_logprobs = [r["avg_logprobs"] for r in responses]

    correctness_log = []
    for (original_q_dict, response, avg_logprob) in zip(q_dicts, response_texts, avg_logprobs):
        original_q_dict[f"{model_nickname}_response"] = response
        original_q_dict["avg_logprob"] = avg_logprob.item()
        num_options = len(original_q_dict["options"])
        model_answer = get_alphabet_choice(response, num_options)
        original_q_dict["selected_choice"] = model_answer
        original_q_dict["is_correct"] = 1 if model_answer == original_q_dict["answer"] else 0
        correctness_log.append(original_q_dict["is_correct"])

    acc = sum(correctness_log)/len(correctness_log)
    return q_dicts, round(acc, 3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_nickname", type=str, required=True)
    parser.add_argument("-d", "--cuda-devices", type=str, required=True)
    parser.add_argument("-n", "--num-runs", type=int, default=1)
    parser.add_argument("-t", "--task", type=str, required=True)
    parser.add_argument("-s", "--split", type=str, default="all")
    parser.add_argument("-ds", "--dataset-suffix", type=str, default="")

    args = parser.parse_args()
    model_nickname = args.model_nickname
    cuda_devices = args.cuda_devices
    task = args.task
    split = args.split
    num_runs = args.num_runs
    ds_suffix = f"_{args.dataset_suffix}" if args.dataset_suffix else ""


    NUM_GPUS = len(cuda_devices.split(","))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    if model_nickname in ["DeepSeekMath", "DpskMath", "QwenMath"]: 
        max_tokens = 4096
    elif model_nickname in ["Gemma", "BioLlama", "Qwen72", "OpenBio", "Llama31"]:
        max_tokens = 4096*2
    else: 
        max_tokens = 4096*4

    model_fullname = AGENT_MAP[model_nickname]
    llm = LLM(
        model = model_fullname,
        download_dir=MODEL_CACHE_DIR,
        tensor_parallel_size=NUM_GPUS,
        trust_remote_code=True,
        max_model_len = max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_fullname, cache_dir=MODEL_CACHE_DIR, trust_remote_code=True)   

    print(f"#### {task} {split} ###")
    q_dicts = read_json(f"./Data/{task}/{split}.json")
    os.makedirs(f"./Model_Responses/{task}-{split}", exist_ok=True)

    for run in range(num_runs):
        L_out, acc = process_questions(q_dicts, model_nickname, task, llm, tokenizer, max_tokens)
        print(f"Final accuracy on split {split}: {acc}")
        write_json(L_out, f"./Model_Responses/{task}-{split}/{model_nickname}.json")
        

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("Script completed. Exiting")
