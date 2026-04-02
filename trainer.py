"""
trainer.py
Full training loop with:
  - Multi-task loss
  - Weighted samples
  - Early stopping
  - Seqeval NER metrics
  - MLflow tracking
  - ONNX export
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from conversation_analyzer import ConversationAnalyzerModel, NER_LABELS
from dataset import build_dataloaders

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Metrics ───────────────────────────────────────────────────
def compute_ner_metrics(
    all_preds: List[List[str]],
    all_labels: List[List[str]],
) -> Dict[str, float]:
    """Compute NER F1/precision/recall using seqeval-style flat comparison."""
    flat_preds  = [p for seq in all_preds  for p in seq]
    flat_labels = [l for seq in all_labels for l in seq]

    # Filter O tags for entity-level metrics
    entity_mask = [l != "O" for l in flat_labels]
    if not any(entity_mask):
        return {"ner_f1": 0.0, "ner_precision": 0.0, "ner_recall": 0.0}

    ep = [p for p, m in zip(flat_preds,  entity_mask) if m]
    el = [l for l, m in zip(flat_labels, entity_mask) if m]

    return {
        "ner_f1":        f1_score(el, ep, average="weighted", zero_division=0),
        "ner_precision": precision_score(el, ep, average="weighted", zero_division=0),
        "ner_recall":    recall_score(el, ep, average="weighted", zero_division=0),
    }


def compute_classification_metrics(
    all_preds: List[int],
    all_labels: List[int],
    prefix: str,
) -> Dict[str, float]:
    return {
        f"{prefix}_accuracy": accuracy_score(all_labels, all_preds),
        f"{prefix}_f1":       f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


# ── One epoch ─────────────────────────────────────────────────
def train_one_epoch(
    model:     ConversationAnalyzerModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids        = batch["input_ids"],
            attention_mask   = batch["attention_mask"],
            ner_labels       = batch["ner_labels"],
            intent_labels    = batch["intent_labels"],
            action_labels    = batch["action_labels"],
            sentiment_labels = batch["sentiment_labels"],
        )
        loss = outputs["loss"]

        # Apply sample weights (mean over batch)
        weights = batch.get("weight")
        if weights is not None:
            loss = (loss * weights.mean()).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model:  ConversationAnalyzerModel,
    loader: DataLoader,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    ner_preds_all:    List[List[str]] = []
    ner_labels_all:   List[List[str]] = []
    intent_preds_all: List[int] = []
    intent_lbls_all:  List[int] = []
    action_preds_all: List[int] = []
    action_lbls_all:  List[int] = []

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(
            input_ids        = batch["input_ids"],
            attention_mask   = batch["attention_mask"],
            ner_labels       = batch["ner_labels"],
            intent_labels    = batch["intent_labels"],
            action_labels    = batch["action_labels"],
            sentiment_labels = batch["sentiment_labels"],
        )

        total_loss += outputs["loss"].item()
        n_batches  += 1

        # NER decode
        ner_batch_preds  = model.predict_ner(outputs["ner_logits"], batch["attention_mask"])
        ner_batch_labels = batch["ner_labels"]

        for b_idx, (preds_seq, attn) in enumerate(
            zip(ner_batch_preds, batch["attention_mask"])
        ):
            pred_strs  = [p[0] for p in preds_seq]
            label_strs = [
                NER_LABELS[ner_batch_labels[b_idx, t].item()]
                for t in range(attn.sum().item())
                if ner_batch_labels[b_idx, t].item() != -100
            ]
            min_len = min(len(pred_strs), len(label_strs))
            ner_preds_all.append(pred_strs[:min_len])
            ner_labels_all.append(label_strs[:min_len])

        # Intent
        intent_preds = outputs["intent_logits"].argmax(dim=-1).cpu().tolist()
        intent_preds_all.extend(intent_preds)
        intent_lbls_all.extend(batch["intent_labels"].cpu().tolist())

        # Action
        action_preds = outputs["action_logits"].argmax(dim=-1).cpu().tolist()
        action_preds_all.extend(action_preds)
        action_lbls_all.extend(batch["action_labels"].cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics  = {}
    metrics.update(compute_ner_metrics(ner_preds_all, ner_labels_all))
    metrics.update(compute_classification_metrics(intent_preds_all, intent_lbls_all, "intent"))
    metrics.update(compute_classification_metrics(action_preds_all,  action_lbls_all, "action"))
    metrics["overall_f1"] = (
        metrics["ner_f1"] * 0.40
        + metrics["intent_f1"] * 0.30
        + metrics["action_f1"] * 0.30
    )
    metrics["loss"] = avg_loss

    return avg_loss, metrics


# ── ONNX export ───────────────────────────────────────────────
def export_to_onnx(
    model:      ConversationAnalyzerModel,
    output_dir: Path,
    max_length: int = 512,
) -> Path:
    model.eval()
    onnx_path = output_dir / "model.onnx"

    dummy_ids  = torch.ones(1, max_length, dtype=torch.long, device=DEVICE)
    dummy_mask = torch.ones(1, max_length, dtype=torch.long, device=DEVICE)

    torch.onnx.export(
        model,
        args   = (dummy_ids, dummy_mask),
        f      = str(onnx_path),
        input_names  = ["input_ids", "attention_mask"],
        output_names = ["ner_logits", "intent_logits", "action_logits", "sentiment_logits"],
        dynamic_axes = {
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "ner_logits":     {0: "batch", 1: "seq"},
            "intent_logits":  {0: "batch"},
            "action_logits":  {0: "batch"},
            "sentiment_logits": {0: "batch"},
        },
        opset_version   = 17,
        do_constant_folding = True,
    )
    logger.info(f"✅ ONNX exported to {onnx_path}")
    return onnx_path


# ── Main training function ────────────────────────────────────
def train_continuous(
    run_id:       str,
    org_id:       str,
    data_dir:     Path,
    model_dir:    Path,
    learning_rate: float = 2e-5,
    epochs:        int   = 3,
    batch_size:    int   = 32,
    warmup_ratio:  float = 0.1,
    patience:      int   = 3,
    pretrained:    str   = "bert-base-multilingual-cased",
) -> Dict[str, float]:
    """
    Fine-tune the model on new data.  Writes best checkpoint + ONNX to
    {model_dir}/{run_id}/ and returns final test metrics.
    """

    output_dir = model_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🚀 Training run: {run_id} | org={org_id} | device={DEVICE}")

    # ── Load data ─────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        train_path = data_dir / "train.json",
        val_path   = data_dir / "validation.json",
        test_path  = data_dir / "test.json",
        tokenizer_name = pretrained,
        batch_size     = batch_size,
        num_workers    = min(4, os.cpu_count() or 1),
    )

    # ── Load or init model ────────────────────────────
    current_model_dir = model_dir / "current"
    if current_model_dir.exists():
        logger.info(f"Loading base model from {current_model_dir}")
        model = ConversationAnalyzerModel.from_pretrained(str(current_model_dir))
    else:
        logger.info(f"Initializing model from config (no weights download): {pretrained}")
        from conversation_analyzer import build_model
        model = build_model(pretrained, load_weights=False)

    model.to(DEVICE)

    # ── Optimizer & scheduler ─────────────────────────
    optimizer   = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps= int(total_steps * warmup_ratio)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── MLflow tracking ───────────────────────────────
    mlflow.set_experiment("lyd-conversation-analyzer")
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params({
            "run_id":       run_id,
            "org_id":       org_id,
            "learning_rate":learning_rate,
            "epochs":       epochs,
            "batch_size":   batch_size,
            "pretrained":   pretrained,
            "train_size":   len(train_loader.dataset),
            "val_size":     len(val_loader.dataset),
        })

        # ── Training loop ─────────────────────────────
        best_val_loss  = float("inf")
        patience_count = 0
        best_epoch     = 0

        for epoch in range(1, epochs + 1):
            t0         = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            val_loss, val_metrics = evaluate(model, val_loader)
            elapsed    = time.time() - t0

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_F1={val_metrics['overall_f1']:.4f} | {elapsed:.1f}s"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss":   val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                },
                step=epoch,
            )

            # ── Early stopping ────────────────────────
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_epoch     = epoch
                patience_count = 0
                model.save_pretrained(str(output_dir))
                logger.info(f"  💾 Checkpoint saved (val_loss={val_loss:.4f})")
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f"  ⏹  Early stopping at epoch {epoch} (best={best_epoch})")
                    break

        # ── Test evaluation ───────────────────────────
        # Reload best checkpoint
        model = ConversationAnalyzerModel.from_pretrained(str(output_dir))
        model.to(DEVICE)

        test_loss, test_metrics = evaluate(model, test_loader)
        logger.info(f"📊 Test | loss={test_loss:.4f} | metrics={test_metrics}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── ONNX export ───────────────────────────────
        onnx_path = export_to_onnx(model, output_dir)
        mlflow.log_artifact(str(onnx_path))

        # ── Save tokenizer alongside model ────────────
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pretrained)
        tok.save_pretrained(str(output_dir / "tokenizer"))

        # ── Log final metrics ─────────────────────────
        result_metrics = {
            "ner_f1":          test_metrics.get("ner_f1", 0.0),
            "ner_precision":   test_metrics.get("ner_precision", 0.0),
            "ner_recall":      test_metrics.get("ner_recall", 0.0),
            "action_f1":       test_metrics.get("action_f1", 0.0),
            "intent_accuracy": test_metrics.get("intent_accuracy", 0.0),
            "relation_f1":     0.0,          # computed separately if relation data present
            "overall_f1":      test_metrics.get("overall_f1", 0.0),
            "loss":            test_loss,
        }
        mlflow.log_metrics(result_metrics)
        mlflow.log_artifact(str(output_dir))

    logger.info(f"Training complete | output={output_dir}")

    # Print JSON metrics as last stdout line (parsed by NestJS)
    print(json.dumps(result_metrics))
    return result_metrics
