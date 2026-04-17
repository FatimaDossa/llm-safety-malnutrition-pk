import json
import os
import csv
from llama_cpp import Llama

MODEL_PATH = "./model/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
PROMPT_DIR = "./prompts"
OUTPUT_FILE = "./outputs/condition_A_base.csv"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4
)
# input as json -> output as csv correct approach 

def load_prompts():
    all_prompts = []
    for file in os.listdir(PROMPT_DIR):
        if file.endswith(".json"):
            with open(os.path.join(PROMPT_DIR, file), "r") as f:
                all_prompts.extend(json.load(f))
    return all_prompts

def generate(prompt):
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>"

    output = llm(
        full_prompt,
        max_tokens=300,
        temperature=0.2
    )
# change here (max_tokens and temperature) incase of long/short output or any modifications 
    return output["choices"][0]["text"].strip()

def run():
    prompts = load_prompts()
    os.makedirs("../outputs", exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["id", "tier", "prompt", "response"])

        for item in prompts:
            print(f"Running {item['id']}...")
            response = generate(item["prompt"])

            writer.writerow([
                item["id"],
                item["tier"],
                item["prompt"],
                response
            ])

    print("Condition A complete.")

if __name__ == "__main__":
    run()