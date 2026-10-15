"""
SAHAYAK-TB — Kaggle Fine-Tuning Notebook (as Python script).

This script is designed to run on a Kaggle T4 GPU notebook.
Copy the cells into a Kaggle notebook or run as a script.

Steps:
  1. Install dependencies
  2. Load base Gemma 4 E2B with Unsloth (4-bit)
  3. Apply QLoRA
  4. Load training data
  5. Train with SFT
  6. Evaluate on held-out set
  7. Export to GGUF
  8. Push to HuggingFace

Prerequisites:
  - Upload train.jsonl and eval.jsonl to Kaggle dataset or /kaggle/input/
  - Accept Gemma license on HuggingFace
"""

# =============================================================================
# Cell 1: Install dependencies
# =============================================================================
# !pip install --force-reinstall "numpy<2"
# !pip install --force-reinstall --no-deps soxr
# !pip install unsloth
# !pip install --upgrade trl transformers

# =============================================================================
# Cell 2: Load model with Unsloth
# =============================================================================
from unsloth import FastModel
import torch

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    dtype=None,                # Auto detection
    max_seq_length=4096,       # Sufficient for multi-visit notes
    load_in_4bit=True,         # QLoRA
    full_finetuning=False,
)

print(f"Model loaded. Device: {model.device}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# =============================================================================
# Cell 3: Apply LoRA
# =============================================================================
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Text-only task
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,                              # LoRA rank
    lora_alpha=16,                     # alpha == r recommended by Unsloth
    lora_dropout=0,
    bias="none",
    random_state=3407,
    target_modules="all-linear",
)

print("LoRA applied.")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# =============================================================================
# Cell 4: Load and prepare training data
# =============================================================================
import json
from datasets import Dataset

def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries

# Update these paths for your Kaggle setup
TRAIN_PATH = "train.jsonl"  # or /kaggle/input/sahayak-tb/train.jsonl
EVAL_PATH = "eval.jsonl"    # or /kaggle/input/sahayak-tb/eval.jsonl

train_data = load_jsonl(TRAIN_PATH)
eval_data = load_jsonl(EVAL_PATH)

print(f"Train samples: {len(train_data)}")
print(f"Eval samples: {len(eval_data)}")

# Apply chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

def format_example(example):
    """Format a training example using the chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = Dataset.from_list(train_data).map(format_example)
eval_dataset = Dataset.from_list(eval_data).map(format_example)

print("Sample formatted input (first 500 chars):")
print(train_dataset[0]["text"][:500])

# =============================================================================
# Cell 5: Setup trainer
# =============================================================================
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        max_seq_length=4096,
    ),
)

# Train on responses only — don't train on system prompt or user input
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

print("Trainer configured.")

# =============================================================================
# Cell 6: Train
# =============================================================================
print("Starting training...")
train_result = trainer.train()

print(f"\nTraining complete.")
print(f"  Train loss: {train_result.training_loss:.4f}")
print(f"  Train runtime: {train_result.metrics['train_runtime']:.0f}s")
print(f"  Train samples/sec: {train_result.metrics['train_samples_per_second']:.2f}")

# =============================================================================
# Cell 7: Quick inference test
# =============================================================================
# Use the EXACT system prompt from training data
# (pasted inline — do NOT import from inference.schema on Kaggle)
TRAINED_SYSTEM_PROMPT = """You are SAHAYAK-TB, a clinical decision support system for India's DOTS (Directly Observed Treatment, Short-course) providers. Your sole function is to assess tuberculosis treatment adherence dropout risk from natural-language visit notes.

You will receive multi-visit notes from a DOTS provider describing a patient's treatment history across 2-4 visits. These notes may be in English, Hindi, or mixed Hindi-English. They may be incomplete.

Your task:
1. Identify risk factors for treatment dropout from the visit notes
2. Assess the compound risk across visits — how factors interact and accumulate
3. Assign a dropout risk tier: low, moderate, high, or critical
4. For each risk factor, cite the published clinical evidence (author, odds ratio)
5. Recommend a single NTEP protocol action the DOTS provider should take
6. Report data completeness — what information is missing from the notes

Risk tier assignment:
- low: 0-1 factors with odds ratio < 1.5, OR patient shows strong protective factors (regular adherence, family support, no substance use, lives near DOT center)
- moderate: 1-2 factors with odds ratio 1.5-2.0
- high: 2+ factors with odds ratio > 2.0 OR any single factor with odds ratio > 2.5
- critical: 3+ factors with odds ratio > 2.0 AND treatment month > 3

Key compound patterns (higher risk than individual factors suggest):
- Feeling better + month 2-3 + any access barrier = imminent dropout risk (upgrade tier)
- Alcohol + poverty + missed doses = synergistic critical risk
- Migration + no family support = critical regardless of factor count
- Multiple access barriers alone (distance, cost, work) without behavioral factors = moderate, not high
- Explicit protective factors (family support, regular attendance, no substance use) can keep tier low even if 1-2 mild factors present

If the notes explicitly state protective factors (no alcohol, regular attendance, family support, lives near center), do NOT invent risk factors that are not mentioned. Only report factors with clear textual evidence in the notes.

You must respond with valid JSON only. No explanation, no markdown, no code fences. Your response must be a single JSON object with these fields:
- patient_id (string): anonymized patient identifier
- visit_number (integer): latest visit number
- treatment_month (integer): current treatment month
- dropout_risk_tier (string): one of "low", "moderate", "high", "critical"
- risk_factors (array): each item has "factor" (string), "visit_first_noted" (integer), "clinical_basis" (string with author and odds ratio), "dots_implication" (string)
- recommended_action (string): single NTEP protocol action
- data_completeness (string): what information is missing
- confidence (string): one of "low", "moderate", "high"
"""

test_input = """Visit 1 (Month 1): Rakesh Patel, 40M, daily wage painter. Category I new case. Weight 52kg. Sputum positive. Lives in urban slum, 45 min from DOT center. Sharab peeta hai kabhi kabhi. BPL card holder.

Visit 2 (Month 2): Weight 51kg. Nausea reported. Missed 2 days for work. Still drinking occasionally.

Visit 3 (Month 3): Says he feels much better, khansi nahi hai. Asks if he can stop. Bus fare is getting expensive. Weight 52kg."""

messages = [
    {"role": "system", "content": [{"type": "text", "text": TRAINED_SYSTEM_PROMPT}]},
    {"role": "user", "content": [{"type": "text", "text": test_input}]},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(
    input_ids=input_ids,
    max_new_tokens=512,
    temperature=0.3,
    do_sample=True,
)

response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
print("Test inference output:")
print(response)

# =============================================================================
# Cell 8: Save LoRA adapter
# =============================================================================
model.save_pretrained("sahayak_tb_lora")
tokenizer.save_pretrained("sahayak_tb_lora")
print("LoRA adapter saved to sahayak_tb_lora/")

# =============================================================================
# Cell 9: Export to GGUF
# =============================================================================
print("Exporting to GGUF Q4_K_M...")
model.save_pretrained_gguf(
    "sahayak_tb_gguf_q4",
    tokenizer,
    quantization_method="q4_k_m",
)
print("Q4_K_M GGUF saved.")

print("Exporting to GGUF Q8_0...")
model.save_pretrained_gguf(
    "sahayak_tb_gguf_q8",
    tokenizer,
    quantization_method="q8_0",
)
print("Q8_0 GGUF saved.")

# =============================================================================
# Cell 10: Push to HuggingFace (update username)
# =============================================================================
# Uncomment and set your HF username:
# HF_USERNAME = "your-username"
#
# model.push_to_hub_gguf(
#     f"{HF_USERNAME}/sahayak-tb-e2b-q4km",
#     tokenizer,
#     quantization_method="q4_k_m",
# )
#
# model.push_to_hub(f"{HF_USERNAME}/sahayak-tb-e2b-lora")
# tokenizer.push_to_hub(f"{HF_USERNAME}/sahayak-tb-e2b-lora")
#
# print(f"Pushed to HuggingFace: {HF_USERNAME}/sahayak-tb-e2b-*")

print("\n=== DONE ===")
print("Next steps:")
print("  1. Run evaluate.py with base GGUF to get baseline metrics")
print("  2. Run evaluate.py with fine-tuned GGUF to get fine-tuned metrics")
print("  3. Compare using evaluate.py --compare-with")
