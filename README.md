# Calibrating Safe and Useful LLMs for Malnutrition Education in Pakistan

**Azkaa Nasir (an08017)** · **Fatima Dossa (fd08024)**  
Habib University — CS 435 · April 2025

---

## Overview

Malnutrition affects **37.6% of children under five in Pakistan**, yet most Large Language Models (LLMs) are not calibrated for local healthcare contexts.

This project evaluates whether **safety-calibrated LLM systems** can reduce:

- Unsafe medical advice
- Hallucinations

**without degrading helpfulness** in malnutrition-related guidance for caregivers in rural Sindh.

---

## Research Question

> Can safety-calibrated LLMs reduce unsafe advice and hallucinations in malnutrition education for caregivers in rural Sindh without degrading helpfulness?

---

## Experimental Design

Four-condition ablation study:

| Condition | System           | Description                                           | Hypothesis                                 |
|-----------|------------------|-------------------------------------------------------|--------------------------------------------|
| A         | Baseline         | Plain LLM, no guardrails, no retrieval                | Highest unsafe advice and hallucinations   |
| B         | Guardrails       | Safety prompt with restrictions and referral triggers | Lower unsafe advice, possible over-refusal |
| C         | RAG              | Retrieval from nutrition knowledge base               | Lower hallucination rate                   |
| D         | RAG + Guardrails | Combined system                                       | Best safety-usefulness tradeoff            |

> ⚠️ Due to time constraints, Conditions C and D may be partially implemented or treated as future work.

---

## Prompt Dataset

- **200 prompts** derived from real CMAM clinical records (Sujawal, Sindh)
- Organized into **3 tiers**:
  - **Tier 1** — Safe / general nutrition queries
  - **Tier 2** — Ambiguous / mild symptoms
  - **Tier 3** — High-risk / medical scenarios

---

## Evaluation Metrics

Each model response is manually annotated using:

| Metric | Name                         | Description                          |
|--------|------------------------------|--------------------------------------|
| UAR    | Unsafe Advice Rate           | Harmful or incorrect guidance        |
| HR     | Hallucination Rate           | Unsupported or fabricated claims     |
| ORR    | Over-Refusal Rate            | Unnecessary refusal to answer        |
| US     | Usefulness Score             | Practical, actionable value          |
| SUT    | Safety–Usefulness Tradeoff   | Composite metric                     |

**Ground truth anchored to:**
- WHO 2023 Wasting Guidelines
- Pakistan National Nutrition Survey (NNS) 2018

---

## Project Structure

```
project/
│
├── src/
│   ├── test_load.py
│   ├── run_fake.py              # baseline condition (A)
│   ├── run_guardrails.py        # guardrails condition (B)
│   └── utils.py                 # API calls
│
├── data/
│   ├── prompts.json
│   ├── outputs.json
│   └── outputs_guardrails.json
│
├── prompts/
│   └── guardrails.txt
│
├── .env
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install requests python-dotenv
```

### 2. Add API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key
```

---

## Running Experiments

### Baseline — Condition A

```bash
python src/run_fake.py
```

Output saved to `data/outputs.json`

---

### Guardrails — Condition B

```bash
python src/run_guardrails.py
```

Output saved to `data/outputs_guardrails.json`

---

## Guardrails Design

Guardrails are implemented via **prompt engineering**, including:

- Diagnosis restrictions
- Referral triggers (e.g., *"consult a doctor"*)
- Safety constraints for medical advice
- Controlled refusal behavior

---

## Responsible AI Framework

| Principle      | Implementation                                    |
|----------------|---------------------------------------------------|
| Safety         | Measured via Unsafe Advice Rate (UAR)             |
| Fairness       | Subgroup analysis (SAM/MAM, rural access, age)    |
| Accountability | Structured annotation pipeline                    |
| Transparency   | Full prompt and output release                    |

---

## Success Criteria

- ≥ 30% reduction in Unsafe Advice Rate
- Decrease in Hallucination Rate with RAG
- ≤ 10–15% increase in Over-Refusal Rate
- Highest SUT score for combined system (Condition D)
- Inter-annotator agreement **κ ≥ 0.70**

---

## Deliverables

- [ ] Annotated dataset (200 prompts)
- [ ] Model output logs (all conditions)
- [ ] Evaluation scripts and scoring pipeline
- [ ] Ablation results table (UAR, HR, ORR, US, SUT)
- [ ] Failure mode taxonomy
- [ ] Fairness analysis
- [ ] Research paper (8–10 pages)

---

## Target Users & Impact

- Caregivers in low-literacy, rural Pakistani settings
- NGOs running CMAM programs in Sindh
- AI practitioners building healthcare systems in the Global South

---

## Future Work

- [ ] Implement full RAG pipeline (Conditions C & D)
- [ ] Expand dataset beyond Sujawal region
- [ ] Integrate multilingual support (Urdu / Sindhi)
- [ ] Conduct real-world deployment evaluation

---

*Habib University · CS 435 · April 2025*
