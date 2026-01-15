import os
from datasets import load_dataset
from code.utils import write_json

ALPHABET = "ABCDEFG"

def main():
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.select(range(30_000))

    updated_data = []
    for idx, example in enumerate(ds):

        new_ex = {}
        new_ex["question_id"] = idx
        new_ex["question"] = example["question"]
        options = []
        options.append(example["opa"])
        options.append(example["opb"])
        options.append(example["opc"])
        options.append(example["opd"])
        new_ex["options"] = options
        new_ex["answer"] = ALPHABET[example["cop"]]
        new_ex["num_choices"] = len(new_ex["options"])
        new_ex["category"] = example["subject_name"]

        updated_data.append(new_ex)
    
    os.makedirs("./Data/MedMCQA", exist_ok=True)
    write_json(updated_data, os.path.join("./Data/MedMCQA", "all.json"))

if __name__ == "__main__":
    main()