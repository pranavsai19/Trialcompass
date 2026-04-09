"""
LoRA fine-tuning for the MS-MARCO cross-encoder on clinical trial pairs.

Base model : cross-encoder/ms-marco-MiniLM-L-6-v2
LoRA config: r=8, alpha=16, target_modules=[query, key, value], dropout=0.1
Data       : data/training_pairs.json (built by build_training_pairs.py)
Output     : models/crossencoder_lora_v1/   (full fine-tuned model)
             models/crossencoder_lora_adapter/  (LoRA adapter weights only)

Why LoRA here?
--------------
The base cross-encoder has ~22M parameters. Full fine-tuning on ~64 pairs
would overfit badly. LoRA injects rank-8 adapters into Q/K/V projections,
adding ~147K trainable parameters (~0.66% of total). This is regularized
enough to learn clinical preference signals without memorizing the tiny
training set.

Run:
  PYTHONPATH=. python src/training/finetune_crossencoder.py

Prerequisites:
  - Run build_training_pairs.py first to produce data/training_pairs.json
  - peft installed: pip install peft
"""

import json
import os
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

TRAINING_PAIRS_PATH  = Path("data/training_pairs.json")
MODEL_OUTPUT_DIR     = Path("models/crossencoder_lora_v1")
ADAPTER_OUTPUT_DIR   = Path("models/crossencoder_lora_adapter")
RESULTS_PATH         = Path("results/lora_training_results.json")

BASE_CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.1
LORA_TARGETS    = ["query", "key", "value"]

TRAIN_RATIO  = 0.80
EPOCHS       = 8
BATCH_SIZE   = 4
WARMUP_STEPS = 4
SEED         = 42


# ---------------------------------------------------------------------------
# Lazy imports — heavy ML imports deferred so `import` of this module is fast
# ---------------------------------------------------------------------------

def _run() -> None:
    import numpy as np
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score
    from scipy.stats import pearsonr
    from torch.utils.data import DataLoader

    from sentence_transformers import CrossEncoder, InputExample
    from peft import LoraConfig, TaskType, get_peft_model

    # ── Load training pairs ──────────────────────────────────────────────

    if not TRAINING_PAIRS_PATH.exists():
        print(f"ERROR: {TRAINING_PAIRS_PATH} not found.")
        print("Run: PYTHONPATH=. python src/training/build_training_pairs.py")
        sys.exit(1)

    with open(TRAINING_PAIRS_PATH) as f:
        all_pairs = json.load(f)

    print(f"Loaded {len(all_pairs)} training pairs "
          f"({sum(p['label'] for p in all_pairs)} positive, "
          f"{sum(1-p['label'] for p in all_pairs)} negative)\n")

    # ── Stratified train/val split ────────────────────────────────────────

    random.seed(SEED)
    positives = [p for p in all_pairs if p["label"] == 1]
    negatives = [p for p in all_pairs if p["label"] == 0]

    random.shuffle(positives)
    random.shuffle(negatives)

    n_pos_train = max(1, int(len(positives) * TRAIN_RATIO))
    n_neg_train = max(1, int(len(negatives) * TRAIN_RATIO))

    train_set = positives[:n_pos_train] + negatives[:n_neg_train]
    val_set   = positives[n_pos_train:] + negatives[n_neg_train:]

    random.shuffle(train_set)
    random.shuffle(val_set)

    print(f"Train: {len(train_set)} pairs "
          f"({sum(p['label'] for p in train_set)} pos, "
          f"{sum(1-p['label'] for p in train_set)} neg)")
    print(f"Val:   {len(val_set)} pairs "
          f"({sum(p['label'] for p in val_set)} pos, "
          f"{sum(1-p['label'] for p in val_set)} neg)\n")

    # ── Load base cross-encoder ───────────────────────────────────────────

    print(f"Loading base model: {BASE_CE_MODEL}", flush=True)
    model = CrossEncoder(BASE_CE_MODEL, num_labels=1, max_length=512)

    # ── Apply LoRA ────────────────────────────────────────────────────────

    print("\nApplying LoRA...", flush=True)

    total_before = sum(p.numel() for p in model.model.parameters())

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    model.model = get_peft_model(model.model, lora_config)

    total_params     = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(
        p.numel() for p in model.model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable (LoRA) : {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen           : {frozen_params:,}\n")

    # ── Prepare DataLoader ────────────────────────────────────────────────

    train_examples = [
        InputExample(texts=[p["query"], p["passage"]], label=float(p["label"]))
        for p in train_set
    ]
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=BATCH_SIZE
    )

    # ── Train ─────────────────────────────────────────────────────────────

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training for {EPOCHS} epochs, batch_size={BATCH_SIZE}, "
          f"warmup_steps={WARMUP_STEPS}", flush=True)
    print(f"Output: {MODEL_OUTPUT_DIR}\n", flush=True)

    model.fit(
        train_dataloader=train_dataloader,
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        output_path=str(MODEL_OUTPUT_DIR),
        show_progress_bar=True,
    )

    print(f"\nModel saved to {MODEL_OUTPUT_DIR}", flush=True)

    # ── Validate ──────────────────────────────────────────────────────────

    print("\nRunning validation...", flush=True)

    val_queries  = [p["query"]   for p in val_set]
    val_passages = [p["passage"] for p in val_set]
    val_labels   = [p["label"]   for p in val_set]

    val_scores = model.predict(list(zip(val_queries, val_passages)))

    # Pearson correlation
    if len(val_labels) > 1 and len(set(val_labels)) > 1:
        pearson_r, pearson_p = pearsonr(val_scores, val_labels)
    else:
        pearson_r, pearson_p = float("nan"), float("nan")

    # Classification metrics at threshold 0.5
    val_preds = [1 if s > 0.5 else 0 for s in val_scores]
    if len(set(val_labels)) > 1:
        val_f1        = float(f1_score(val_labels, val_preds, zero_division=0))
        val_precision = float(precision_score(val_labels, val_preds, zero_division=0))
        val_recall    = float(recall_score(val_labels, val_preds, zero_division=0))
    else:
        # Only one class in val — can't compute meaningful F1
        val_f1 = val_precision = val_recall = float("nan")

    val_accuracy = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)

    print("\nValidation results:")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-'*32}")
    print(f"  {'Pearson r':<20} {pearson_r:>10.4f}")
    print(f"  {'Accuracy':<20} {val_accuracy:>10.4f}")
    print(f"  {'Precision':<20} {val_precision:>10.4f}")
    print(f"  {'Recall':<20} {val_recall:>10.4f}")
    print(f"  {'F1':<20} {val_f1:>10.4f}")

    # Per-example detail
    print(f"\n  {'#':<4} {'Label':>6} {'Score':>8} {'Pred':>6} {'NCT ID':<15}")
    print(f"  {'-'*45}")
    for i, (lbl, score, pred, p) in enumerate(
        zip(val_labels, val_scores, val_preds, val_set), 1
    ):
        mark = "✓" if pred == lbl else "✗"
        print(f"  {i:<4} {lbl:>6} {score:>8.4f} {pred:>6}  {mark}  {p['nct_id']}")

    # ── Save LoRA adapter separately ──────────────────────────────────────

    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save just the LoRA adapter weights (small — only the delta parameters)
    model.model.save_pretrained(str(ADAPTER_OUTPUT_DIR))
    print(f"\nLoRA adapter saved to {ADAPTER_OUTPUT_DIR}")

    training_info = {
        "base_model":      BASE_CE_MODEL,
        "lora_rank":       LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "lora_targets":    LORA_TARGETS,
        "lora_dropout":    LORA_DROPOUT,
        "n_train_pairs":   len(train_set),
        "n_val_pairs":     len(val_set),
        "n_positive":      sum(p["label"] for p in all_pairs),
        "n_negative":      sum(1 - p["label"] for p in all_pairs),
        "epochs":          EPOCHS,
        "batch_size":      BATCH_SIZE,
        "warmup_steps":    WARMUP_STEPS,
        "training_date":   str(date.today()),
        "total_params":    total_params,
        "trainable_params": trainable_params,
        "val_accuracy":    round(val_accuracy, 4),
        "val_precision":   round(val_precision, 4) if not (val_precision != val_precision) else None,
        "val_recall":      round(val_recall, 4) if not (val_recall != val_recall) else None,
        "val_f1":          round(val_f1, 4) if not (val_f1 != val_f1) else None,
        "val_pearson_r":   round(pearson_r, 4) if not (pearson_r != pearson_r) else None,
    }

    training_info_path = ADAPTER_OUTPUT_DIR / "training_info.json"
    with open(training_info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved to {training_info_path}")

    # ── Save full results ─────────────────────────────────────────────────

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_results = {
        "training_config": training_info,
        "val_per_example": [
            {
                "nct_id":  p["nct_id"],
                "query":   p["query"][:60],
                "label":   lbl,
                "score":   float(score),
                "pred":    pred,
                "correct": pred == lbl,
            }
            for p, lbl, score, pred in zip(val_set, val_labels, val_scores, val_preds)
        ],
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"Full results saved to {RESULTS_PATH}")

    print("\n" + "=" * 60)
    print(f"FINE-TUNING COMPLETE")
    print(f"  Model : {MODEL_OUTPUT_DIR}")
    print(f"  Adapter: {ADAPTER_OUTPUT_DIR}")
    print(f"  Val F1={val_f1:.4f}  Acc={val_accuracy:.4f}  Pearson={pearson_r:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run()
