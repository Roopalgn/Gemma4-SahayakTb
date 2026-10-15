# SAHAYAK-TB

**Adherence Dropout Risk Intelligence for India's DOTS Providers**

> Fine-tuned Gemma 4 E2B that assesses TB treatment dropout risk from natural-language visit notes — citation-backed, fully offline, CPU-deployable.

| | |
|---|---|
| 🌐 **Live Demo** | [huggingface.co/spaces/Roopalgn/sahayak-tb-demo](https://huggingface.co/spaces/Roopalgn/sahayak-tb-demo) |
| 🤗 **Model Weights** | [Roopalgn/sahayak-tb-gemma4-e2b-gguf](https://huggingface.co/Roopalgn/sahayak-tb-gemma4-e2b-gguf) |
| 🏆 **Tracks** | Health & Sciences · llama.cpp Prize |

---

## The Problem

India accounts for 32% of the world's drug-resistant TB cases. **DR-TB doesn't appear spontaneously — it is created when patients stop treatment early.**

India's NTEP (National TB Elimination Programme) operates 25,000+ DOTS treatment centres. Providers monitor patients through 6–20 month drug regimens and record visit notes in NIKSHAY. The failure point is not diagnosis — it is treatment completion.

**The two-stage collapse:**
1. **Silent drift (month 2–3):** Patient feels better. Side effects accumulate. Travel costs compound. No single visit looks alarming — but the *combination* of signals across visits represents 70%+ dropout probability.
2. **Hard dropout:** Patient stops attending. NIKSHAY flags a missed dose. By then, resistance may be established.

**The intervention window is Stage 1. No deployed system catches it.**

Existing tools fail because:
- **NIKSHAY's built-in rules** fire *after* two missed doses — reactive, not predictive
- **Structured ML classifiers** require clean tabular input; DOTS providers write conversational notes in Hindi-English
- **General LLMs via API** send patient data offsite, violating India's DPDP Act, and have no calibration to NTEP risk factors

---

## The Solution

SAHAYAK-TB takes unstructured, natural-language multi-visit notes from a DOTS provider and outputs a structured, evidence-grounded dropout risk assessment — **fully offline, on CPU, in under 90 seconds**.

**One input:** Conversational notes from 2–4 patient visits (English, Hindi, or mixed)

**One output:**
```json
{
  "dropout_risk_tier": "high",
  "risk_factors": [
    {
      "factor": "Daily wage earner — treatment costs causing work absence",
      "clinical_basis": "Kumar et al. 2024 (OR 1.7–2.3, PMID 39067959)",
      "dots_implication": "Refer for Nikshay Poshan Yojana nutritional support"
    },
    {
      "factor": "Migrant worker, no fixed address",
      "clinical_basis": "Vijay et al. 2010 (OR 2.0–2.8, PMID 20386611)",
      "dots_implication": "Initiate NIKSHAY transfer and mobile follow-up"
    }
  ],
  "recommended_action": "nikshay_transfer_mobile_followup",
  "confidence": "high"
}
```

---

## How Gemma 4 Solves This

### Why Gemma 4 E2B specifically

| Requirement | How Gemma 4 E2B meets it |
|-------------|--------------------------|
| Hindi-English code-switching in notes | Gemma 4's multilingual pretraining handles mixed-language input natively |
| No patient data leaving the device | GGUF via llama.cpp — fully local, zero API calls |
| DPDP Act compliance | Offline inference; no cloud dependency |
| Deployable on DOTS centre hardware | Q4_K_M quantization = 3.43 GB, runs on any laptop CPU |
| $0 deployment cost | Free Kaggle T4 fine-tuning + llama.cpp CPU inference |
| Structured clinical output | Gemma 4's instruction-following + schema-enforced JSON via llama.cpp |

### Fine-Tuning with Unsloth

Base: `unsloth/gemma-4-E2B-it` (4-bit QLoRA via Unsloth FastModel)

| Parameter | Value |
|-----------|-------|
| Method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Target modules | all-linear |
| Batch size | 1 (grad accum 4) |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Optimizer | AdamW-8bit |
| Trainable parameters | 29.8M / 5.15B (0.58%) |
| Platform | Kaggle T4 GPU (free tier, $0) |
| Training time | 33 minutes |

**Key training choice:** `train_on_responses_only` — the model learns only to generate the JSON risk assessment, not to reproduce the system prompt or patient notes. This keeps the fine-tune focused and avoids catastrophic forgetting.

### Training Data Pipeline

500 training + 100 evaluation examples, synthetically generated from **15 published risk factors** with verified odds ratios and PMIDs:

| Factor | Odds Ratio | Source |
|--------|-----------|--------|
| Alcohol use | 2.9–3.5 | Santha et al. 2002 (PMID 12234133) |
| Retreatment case | 2.0–3.0 | Sarangi & Dutt 2014 (PMID 25241570) |
| Migration / no fixed address | 2.0–2.8 | Vijay et al. 2010 (PMID 20386611) |
| Side effects (intensive phase) | 1.8–2.5 | Kumar et al. 2024 (PMID 39067959) |
| HIV co-infection | 1.8–2.5 | Reddy Satti et al. 2016 |
| Daily wage / poverty | 1.7–2.3 | Kumar et al. 2024 (PMID 39067959) |
| *(+9 more in `data/risk_factors.json`)* | | |

Each factor has Hindi and English natural-language variants (e.g., `"sharab ki aadat hai"` / `"patient drinks regularly"`). Ground truth labels are derived deterministically from factor count × odds ratio × compound interaction rules from WHO guidelines.

No patient-identifiable data was used. Full methodology in `data/generate_training_data.py`.

---

## Technical Architecture

```
DOTS Provider Visit Notes (Hindi/English/mixed)
              |
              v
  +-----------------------+
  |   System Prompt        |  <- NTEP risk rules + compound patterns
  |   + Visit Notes        |     (inference/schema.py)
  +-----------------------+
              |
              v
  +-----------------------+
  |  Fine-tuned Gemma 4   |  <- QLoRA on 15 risk factors
  |  E2B (Q4_K_M GGUF)    |     llama-cpp-python, CPU only
  |  Schema-enforced JSON  |     n_ctx=4096, temp=0.0
  +-----------------------+
              |
              v
  +-----------------------+
  |   Post-Processing      |  <- 3 layers:
  |   (postprocess.py)     |     1. Citation grounding (real PMIDs)
  +-----------------------+     2. Tier correction (deterministic rules)
              |                  3. Factor deduplication (fuzzy match)
              v
  Risk Tier + Cited Factors + NTEP Action
```

### The Post-Processing Layer

Three deterministic safety layers on top of the model:

1. **Citation grounding:** Maps every model-generated factor to one of 15 verified ground-truth entries with real PMIDs and odds ratios. If no match above confidence threshold, the factor is flagged, not silently presented.

2. **Tier correction:** Deterministic override based on factor count × odds ratio × compound patterns (e.g., `alcohol + poverty + month > 3 = critical regardless of individual factors`).

3. **Factor deduplication:** Fuzzy keyword matching merges semantically equivalent factors the model may express differently across visits.

### Why llama.cpp (Special Technology Track)

This project qualifies for the **llama.cpp Prize** — best innovative use of Gemma 4 on resource-constrained hardware:

- `response_format={"type": "json_object", "schema": RISK_ASSESSMENT_SCHEMA}` enforces the full output schema at the token sampling level — no JSON repair needed
- `n_gpu_layers=0` — intentional CPU-only inference matching the deployment reality of rural DOTS centres
- `temperature=0.0` — deterministic output for a clinical decision support tool
- `n_ctx=4096` — fits 4 full visits with system prompt in one context window

---

## Evaluation Results

Evaluated on a 20-sample held-out set covering all four risk tiers.

### Full Pipeline (Model + Post-Processing + Schema Enforcement)

| Metric | Value |
|--------|-------|
| **Tier accuracy** | **90%** (18/20) |
| **Schema compliance** | **100%** (20/20) |
| **High + critical recall** | **100%** (zero under-triage) |
| Latency range (CPU) | 24–88 seconds |
| Error type | Both errors over-predict by one tier |

### Raw Model Only (no post-processing)

| Metric | Value |
|--------|-------|
| Tier accuracy | 60% (12/20) |
| Schema compliance | 100% (20/20) |

The post-processing layer (citation grounding + tier correction + factor dedup) contributes a 30-point accuracy improvement. This is the intended design — the model provides flexible clinical reasoning, and the deterministic safety layer enforces calibrated risk boundaries.

### Error Analysis

| Case | Expected | Predicted | Direction |
|------|----------|-----------|-----------|
| #6 | high | critical | Conservative (safe) |
| #15 | moderate | high | Conservative (safe) |

No high-risk or critical patient was ever classified as low risk. Both errors are conservative over-predictions — the clinically safe failure mode. Per WHO screening guidelines, sensitivity is prioritized over specificity in triage instruments.

### Training Convergence

| Metric | Value |
|--------|-------|
| Initial loss | 3.04 |
| Final loss | 0.0115 |
| Total steps | 375 |

---

## Real-World Impact

- **Target:** 25,000+ DOTS providers across India's NTEP network
- **Intervention window:** Flags at-risk patients 2–4 weeks before hard dropout
- **Privacy:** Fully offline — zero patient data leaves the device
- **Language:** Handles English, Hindi, and code-mixed notes natively
- **Cost:** Zero recurring cost — runs on existing centre hardware
- **Scale:** Same model architecture applies to any DOTS programme globally (WHO framework)

If SAHAYAK-TB prevents 1% of India's ~400,000 annual TB treatment dropouts, that is 4,000 patients completing treatment instead of potentially developing MDR-TB — each MDR-TB case costs Rs 5–10 lakhs to treat vs Rs 5,000 for first-line completion.

---

## Published Artifacts

| Artifact | Link |
|----------|------|
| GGUF weights (Q4_K_M, 3.43 GB) | [Roopalgn/sahayak-tb-gemma4-e2b-gguf](https://huggingface.co/Roopalgn/sahayak-tb-gemma4-e2b-gguf) |
| Live demo | [Roopalgn/sahayak-tb-demo (HF Spaces)](https://huggingface.co/spaces/Roopalgn/sahayak-tb-demo) |

---

## Project Structure

```
gemma4good/
├── data/
│   ├── risk_factors.json           # 15 risk factors: OR, PMIDs, NTEP actions, Hindi variants
│   ├── ntep_actions.json           # NTEP protocol action set
│   ├── generate_training_data.py   # Synthetic data generator (rule-based)
│   ├── train.jsonl                 # 500 training examples
│   └── eval.jsonl                  # 100 evaluation examples
├── training/
│   ├── finetune.py                 # Unsloth QLoRA fine-tuning (Kaggle T4)
│   ├── evaluate.py                 # Base vs fine-tuned comparison
│   ├── eval_quick.py               # Quick eval on held-out set
│   ├── export_gguf.py              # GGUF export utility
│   └── verify_no_regression.py     # Regression test suite
├── inference/
│   ├── schema.py                   # JSON schema + system prompt + validation
│   ├── postprocess.py              # Citation grounding, tier correction, dedup
│   └── benchmark.py                # llama.cpp CPU latency benchmark
├── demo/
│   ├── app.py                      # FastAPI backend + background model loading
│   ├── Dockerfile                  # HuggingFace Spaces Docker deployment
│   ├── requirements.txt
│   └── static/                     # Frontend (HTML/CSS/JS — no framework)
│       ├── index.html
│       ├── style.css
│       └── app.js
└── results/
    └── eval_20_results.json        # Per-sample eval: tier, match, latency
```

---

## Reproduce

### Generate training data
```bash
cd data/
python generate_training_data.py --mode rule --count 500 --output train.jsonl
python generate_training_data.py --mode rule --count 100 --output eval.jsonl --seed 9999
```

### Fine-tune on Kaggle
Upload `data/train.jsonl` and `data/eval.jsonl` to a Kaggle dataset, then run `training/finetune.py` cells in a T4 GPU notebook (free tier, ~33 min).

### Run demo locally
```bash
pip install fastapi uvicorn llama-cpp-python huggingface_hub
cd demo/
SAHAYAK_MODEL_PATH=/path/to/gemma-4-e2b-it.Q4_K_M.gguf python app.py
# Open http://localhost:7860
```

### Run evaluation
```bash
cd training/
python eval_quick.py --model-path /path/to/model.gguf --eval-data ../data/eval.jsonl
```

---

## Data Sources

All training data derived from publicly verifiable sources. No patient-identifiable data used.

- WHO Treatment of Tuberculosis Guidelines (4th ed.)
- WHO Companion Handbook for DR-TB Programmatic Management
- WHO Adherence to Long-Term Therapies: Evidence for Action
- NTEP/RNTCP DOTS provider training manuals (MoHFW India)
- Vijay et al. 2010 — PLOS ONE (PMID 20386611)
- Santha et al. 2002 — IJTLD (PMID 12234133)
- Kumar et al. 2024 — IJT (PMID 39067959)
- Sarangi & Dutt 2014 (PMID 25241570)
- Zhou et al. 2020 — AJTMH (PMID 32618243)
- Chauhan et al. 2024 (PMID 38812804)

---

## License

CC-BY 4.0
