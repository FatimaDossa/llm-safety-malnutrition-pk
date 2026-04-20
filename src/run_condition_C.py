import json
import os
import csv
from llama_cpp import Llama
from retriever_chroma import retrieve

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = "./model/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
PROMPT_DIR = "./prompts"
OUTPUT_FILE = "./outputs/condition_C_rag.csv"

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4
)

# ─────────────────────────────────────────
# LOAD PROMPTS
# ─────────────────────────────────────────
def load_prompts():
    prompts = []
    for file in os.listdir(PROMPT_DIR):
        if file.endswith(".json"):
            with open(os.path.join(PROMPT_DIR, file), "r", encoding="utf-8") as f:
                prompts.extend(json.load(f))
    return prompts

# ─────────────────────────────────────────
# GENERATION WITH RAG
# ─────────────────────────────────────────
def generate(prompt):
    # Retrieve top-k relevant chunks
    retrieved_docs = retrieve(prompt, top_k=3)

    # Build structured context
    context = ""
    sources_list = []

    print("\nRetrieved Evidence:")
    for r in retrieved_docs:
        context += (
    f"SOURCE: {r['source']}\n"
    f"TOPIC: {r['topic']}\n"
    f"TEXT: {r['text']}\n\n"
)
        sources_list.append(r["source"])

        # Debug print (VERY useful for paper + debugging)
        print("----")
        print("Source:", r["source"])
        print("Topic:", r["topic"])
        print("Text:", r["text"][:200])

    # Final prompt (NO guardrails in Condition C)
    full_prompt = f"""
<|system|>
You are a helpful assistant providing child nutrition guidance.

Use ONLY the provided context to answer.
If the answer is not in the context, say you are unsure.

<|context>
{context}

<|user|>
{prompt}

<|assistant|>
"""

    output = llm(
        full_prompt,
        max_tokens=300,
        temperature=0.2
    )

    response = output["choices"][0]["text"].strip()

    return response, "; ".join(sources_list)

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def main():
    os.makedirs("./outputs", exist_ok=True)

    prompts = load_prompts()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Added retrieval column for analysis
        writer.writerow([
            "id",
            "tier",
            "prompt",
            "response",
            "retrieved_sources"
        ])

        for item in prompts:
            print("\nRunning:", item["id"])

            response, sources = generate(item["prompt"])

            writer.writerow([
                item["id"],
                item["tier"],
                item["prompt"],
                response,
                sources
            ])

    print("\nDONE Condition C (RAG)")

# ─────────────────────────────────────────
if __name__ == "__main__":
    main()