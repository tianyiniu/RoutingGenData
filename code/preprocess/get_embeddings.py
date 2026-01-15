import os
import torch
from tqdm import tqdm
from vllm import LLM
from code.utils import MODEL_CACHE_DIR, read_json
import argparse


def format_question(q_dict):
    question = q_dict["question"]
    output = f"{question}\n\n"
    if not output: 
        raise ValueError("At least one of question, keywords, stepback must be selected")
    return output


def get_embeddings(model, queries, num_cuda): 
    outputs = model.embed(queries)
    embeddings = [o.outputs.embedding for o in outputs]
    return embeddings 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", type=str)
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-d", "--cuda-devices", type=str, default="7")
    args = parser.parse_args()

    CUDA = args.cuda_devices
    TASK = args.task
    split = args.split

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
    tp_size = len(CUDA.split(","))
    model = LLM(
        model="Qwen/Qwen3-Embedding-8B",
        task="embed",
        download_dir=MODEL_CACHE_DIR,
        tensor_parallel_size=tp_size,
        max_model_len=4096*4,
        enable_chunked_prefill=False,
        enforce_eager=True,
    )

    split = "train"
    print(f"Reading: ./Data/{TASK}/{split}.json")
    os.makedirs("./artifacts", exist_ok=True)
    save_path = f"./artifacts/{TASK}_only_question_embeddings_early.pt"
    print(f"Saving to: {save_path}")

    questions = read_json(f"./Data/{TASK}/{split}.json")

    # Build query texts
    question_queries, ids = [], []
    for ex in tqdm(questions):
        full_question_text = format_question(ex)
        question_queries.append(full_question_text)
        ids.append(ex["question_id"])
        assert full_question_text


    embeddings = get_embeddings(model, question_queries, tp_size)
    embeddings_dict = {
        qid: torch.tensor(embed, dtype=torch.float32) for qid, embed in zip(ids, embeddings)
    }  

    torch.save(embeddings_dict, save_path)
    print(f"Saved embeddings to {save_path}")