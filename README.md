# Calibrating Safe and Useful LLMs for Malnutrition Education in Pakistan

**Azkaa Nasir · Fatima Dossa**
Habib University · CS 435 · April 2026

---

## Overview

This project evaluates how safety calibration techniques affect the performance of Large Language Models (LLMs) in **malnutrition education for caregivers in rural Pakistan**.

We conduct a controlled **ablation study** to measure how:
- safety guardrails
- retrieval-augmented generation (RAG)

impact **unsafe advice, hallucinations, and usefulness**.

---

## Research Question

Can safety-calibrated LLMs reduce unsafe advice and hallucinations in malnutrition education without degrading helpfulness?

---

## Methodology

We evaluate a locally deployed LLaMA 3 (3B Instruct) across four conditions:

### Condition A — Base Model
- No guardrails
- No retrieval
- Baseline behavior

### Condition B — Guardrails
- System prompt with:
  - safety constraints
  - referral triggers
  - uncertainty handling

### Condition C — RAG (Retrieval-Augmented Generation)
- Retrieves relevant knowledge from:
  - WHO 2023 Wasting Guidelines
  - Pakistan National Nutrition Survey (2018)
- Uses:
  - chunking + embedding (sentence-transformers)
  - top-k semantic retrieval (ChromaDB)

### Condition D — RAG + Guardrails
- Combines retrieval + safety constraints
- Designed to optimize safety–usefulness tradeoff

---

## Dataset

- 200 prompts derived from Sujawal CMAM clinical records
- Stratified into:
  - Tier 1 (general knowledge)
  - Tier 2 (moderate risk)
  - Tier 3 (emergency scenarios)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **UAR** | Unsafe Advice Rate |
| **HR** | Hallucination Rate |
| **ORR** | Over-Refusal Rate |
| **US** | Usefulness Score |

---

## Project Structure

    .
    ├── model/                     # Local LLaMA GGUF model (not tracked)
    ├── prompts/                   # JSON prompt dataset (tier1, tier2, tier3)
    ├── src/
    │   ├── run_condition_A.py
    │   ├── run_condition_B.py
    │   ├── run_condition_C.py
    │   ├── run_condition_D.py
    │   ├── retriever_chroma.py
    │   └── build_kb.py
    ├── chroma_kb/                 # Vector database (generated)
    ├── outputs/                   # CSV outputs for each condition
    ├── .gitignore
    └── README.md

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd llm-safety-malnutrition-pk
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model

Download and place the following file:

    Llama-3.2-3B-Instruct-Q4_K_M.gguf

inside:

    ./model/

---

## Running Experiments

### Condition A — Base

```bash
python src/run_condition_A.py
```

### Condition B — Guardrails

```bash
python src/run_condition_B.py
```

### Condition C — RAG

First build the knowledge base:

```bash
python src/build_kb.py
```

Then run:

```bash
python src/run_condition_C.py
```

### Condition D — RAG + Guardrails

```bash
python src/run_condition_D.py
```

---

## RAG Pipeline

1. Documents loaded (WHO, NNS, MUAC charts)
2. Text split into chunks
3. Chunks embedded using sentence-transformers
4. Stored in ChromaDB vector database
5. At runtime:
   - query → embedding
   - cosine similarity search
   - top-k relevant chunks retrieved
   - Retrieved context injected into prompt

---

## Outputs

    outputs/
    ├── condition_A_base.csv
    ├── condition_B_guardrails.csv
    ├── condition_C_rag.csv
    └── condition_D_rag_guardrails.csv

Each CSV contains:

    id | tier | prompt | response | retrieved_sources

---

## Key Findings

- Base LLMs produce unsafe and hallucinated medical advice
- Guardrails reduce unsafe outputs but increase over-refusal
- RAG improves factual grounding but not reasoning
- RAG + Guardrails provides the best safety–usefulness balance

---

## Future Work

- Expand localized clinical knowledge base
- Optimize RAG (chunk size, retrieval strategy)
- Improve annotation reliability (Cohen's κ)
- Validate with domain experts and real users

---

## Reproducibility

- Fully local pipeline (no API dependency)
- Fixed prompts and deterministic setup
- Open dataset and outputs

---

## License

For academic and research use.
