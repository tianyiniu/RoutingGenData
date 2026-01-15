import os
from datasets import load_dataset
from code.utils import write_json

def main():
    ds = load_dataset("m-a-p/SuperGPQA", split="train")

    updated_data = []
    for idx, example in enumerate(ds):
        example["question_id"] = idx
        example["category"] = example["discipline"]
        del example["discipline"]

        example["answer"] = example["answer_letter"]
        del example["answer_letter"]
        example["num_choices"] = len(example["options"])
        updated_data.append(example)
    
    os.makedirs("./Data/SuperGPQA", exist_ok=True)
    write_json(updated_data, os.path.join("./Data/SuperGPQA", "all.json"))

if __name__ == "__main__":
    main()