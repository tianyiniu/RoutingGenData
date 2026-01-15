import os
from datasets import load_dataset
from code.utils import write_json

def main():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    updated_data = []
    for idx, example in enumerate(ds):
        del example["answer_index"]
        del example["cot_content"]
        example["num_choices"] = len(example["options"])
        updated_data.append(example)
    
    os.makedirs(".Data/MMLU_Pro", exist_ok=True)
    write_json(updated_data, os.path.join(".Data/MMLU_Pro", "all.json"))

if __name__ == "__main__":
    main()