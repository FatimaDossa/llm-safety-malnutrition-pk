# Calibrating Safe and Useful LLMs for Malnutrition Education in Pakistan

**Azkaa Nasir (an08017)** · **Fatima Dossa (fd08024)**  
Habib University — CS 435 · April 2026

---

## Overview

This project evaluates how safety guardrails affect the performance of a local Large Language Model in generating health-related guidance for caregivers in rural Pakistan. The study compares a base model against a safety-calibrated version using structured prompts across multiple clinical tiers derived from real-world malnutrition contexts.

The goal is to analyze whether safety interventions reduce unsafe advice and hallucinations without significantly degrading helpfulness.

---

## Model

This project uses:

- **Llama 3.2 3B Instruct** (quantized GGUF version)

The model is run locally using `llama-cpp-python` for reproducible and low-resource inference.

---

## Research Objective

> Can safety-calibrated LLMs reduce unsafe advice and hallucinations in malnutrition education for caregivers in rural Sindh without degrading helpfulness?

---

## Experimental Setup

The evaluation is conducted using two conditions:

| Condition | Description |
|-----------|-------------|
| A         | Base model with no system prompt or safety constraints |
| B         | Base model with safety guardrails applied through a system prompt |

Each condition is evaluated on identical prompts across multiple tiers of medical/nutrition-related scenarios.

---

## Evaluation Metrics

| Metric | Name                           | Description                        |
|--------|--------------------------------|------------------------------------|
| UAR    | Unsafe Advice Rate             | Harmful or incorrect guidance      |
| HR     | Hallucination Rate             | Unsupported or fabricated claims   |
| ORR    | Over-Refusal Rate              | Unnecessary refusal to answer      |
| US     | Usefulness Score               | Practical, actionable value        |
| SUT    | Safety–Usefulness Tradeoff     | Composite metric                   |

---

## Dataset

The dataset consists of structured prompts derived from malnutrition-related scenarios in Pakistan, categorized into tiers representing increasing sensitivity and risk levels.

- **Tier 1** — Safe / general nutrition queries
- **Tier 2** — Ambiguous / mild symptoms
- **Tier 3** — High-risk / medical scenarios

---

## Project Structure

```
project/
│
├── src/
│   ├── run_condition_A.py       # baseline condition (A)
│   ├── run_condition_B.py       # guardrails condition (B)
│   └── utils.py                 # shared utilities
│
├── prompts/
│   └── guardrails.txt
│
├── outputs/
│
├── model/                       # excluded from repository
├── venv/                        # excluded from repository
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repo-url>
cd genai_project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model file

Place the model file at:

```
model/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Running the Experiment

**Condition A** — Base model, no guardrails:

```bash
python src/run_condition_A.py
```

**Condition B** — Base model with safety guardrails:

```bash
python src/run_condition_B.py
```

Outputs are saved to the `outputs/` directory.

---

## Notes

- The model runs locally and does not require external APIs
- The system is designed for reproducibility and low-resource environments
- Inference speed depends on CPU performance due to quantized local execution

---

## Responsible AI Framework

| Principle      | Implementation                                    |
|----------------|---------------------------------------------------|
| Safety         | Measured via Unsafe Advice Rate (UAR)             |
| Fairness       | Subgroup analysis (SAM/MAM, rural access, age)    |
| Accountability | Structured annotation pipeline                    |
| Transparency   | Full prompt and output release                    |

---

## Target Users & Impact

- Caregivers in low-literacy, rural Pakistani settings
- NGOs running CMAM programs in Sindh
- AI practitioners building healthcare systems in the Global South

---

## Future Work

- [ ] Implement full RAG pipeline
- [ ] Expand dataset beyond Sujawal region
- [ ] Integrate multilingual support (Urdu / Sindhi)
- [ ] Conduct real-world deployment evaluation

---

## License

This project is intended for academic use only.

---

*Habib University · CS 435 · April 2026*
