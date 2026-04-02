"""
dataset.py
Utility to build PyTorch DataLoaders for the multi-task conversation analyzer.

Expected JSON formats (both supported):
  1) JSON array:    [ {sample}, {sample}, ... ]
  2) JSON lines:    {"sample": ...}\n{"sample": ...}\n

Each sample may contain:
  - text: str (required)
  - entities: list[dict] (optional)
      Supported keys:
        - type/label: "person"|"date"|"project"|"amount"|"org" (or already "PERSON"...)
        - value/text: substring to match in text
        - start/end: character offsets (preferred when available)
  - intent: str (optional)  -> one of conversation_analyzer.INTENT_LABELS
  - actions: list[dict] or str (optional)
      - if list: first item's "action" is used
      - if str: used directly
  - sentiment: float (optional) in [-1, 1]
  - weight: float (optional) sample weight
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from conversation_analyzer import ACTION_LABELS, INTENT_LABELS, NER_LABELS


_ENTITY_TYPE_TO_SUFFIX = {
    "person": "PERSON",
    "date": "DATE",
    "project": "PROJECT",
    "amount": "AMOUNT",
    "org": "ORG",
    "organization": "ORG",
}


def _read_samples(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a JSON array")
        return [x for x in data if isinstance(x, dict)]

    # JSONL fallback
    samples: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            samples.append(obj)
    return samples


def _norm_entity_type(entity: Dict[str, Any]) -> Optional[str]:
    t = entity.get("type") or entity.get("label")
    if not t:
        return None
    t = str(t).strip()
    if not t:
        return None
    low = t.lower()
    if low in _ENTITY_TYPE_TO_SUFFIX:
        return _ENTITY_TYPE_TO_SUFFIX[low]
    # allow already-normalized suffixes (PERSON/DATE/...)
    up = t.upper()
    if up in {"PERSON", "DATE", "PROJECT", "AMOUNT", "ORG"}:
        return up
    return None


def _find_span(text: str, value: str) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    idx = text.lower().find(value.lower())
    if idx < 0:
        return None
    return idx, idx + len(value)


def _collect_entity_spans(text: str, entities: Any) -> List[Tuple[int, int, str]]:
    if not entities:
        return []
    if not isinstance(entities, list):
        return []

    spans: List[Tuple[int, int, str]] = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        suffix = _norm_entity_type(ent)
        if not suffix:
            continue

        start = ent.get("start")
        end = ent.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
            spans.append((start, end, suffix))
            continue

        val = ent.get("value") or ent.get("text")
        if isinstance(val, str):
            span = _find_span(text, val.strip())
            if span:
                spans.append((span[0], span[1], suffix))

    # stable order: earlier spans first (helps deterministic labeling on overlaps)
    spans.sort(key=lambda x: (x[0], x[1]))
    return spans


def _intent_id(intent: Any) -> int:
    if not isinstance(intent, str) or not intent.strip():
        return 0
    intent = intent.strip()
    try:
        return INTENT_LABELS.index(intent)
    except ValueError:
        return 0


def _action_id(actions: Any) -> int:
    if isinstance(actions, str) and actions.strip():
        action = actions.strip()
    elif isinstance(actions, list) and actions:
        first = actions[0]
        if isinstance(first, dict) and isinstance(first.get("action"), str):
            action = first["action"].strip()
        else:
            action = "none"
    else:
        action = "none"

    try:
        return ACTION_LABELS.index(action)
    except ValueError:
        return 0


@dataclass(frozen=True)
class DatasetConfig:
    tokenizer_name: str
    max_length: int = 128


class ConversationDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], cfg: DatasetConfig):
        self.samples = samples
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
        self.ner_label_to_id = {lab: i for i, lab in enumerate(NER_LABELS)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = str(sample.get("text") or "")

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.cfg.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        offsets: List[Tuple[int, int]] = [tuple(x) for x in enc.pop("offset_mapping")]
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        # NER labeling
        spans = _collect_entity_spans(text, sample.get("entities"))
        ner_ids = [-100] * len(offsets)
        prev_entity: Optional[Tuple[int, int, str]] = None

        for i, (s, e) in enumerate(offsets):
            if attention_mask[i].item() == 0:
                break
            if s == e:  # special tokens typically map to (0,0)
                ner_ids[i] = -100
                continue

            label = "O"
            matched = None
            for (es, ee, suf) in spans:
                if s < ee and e > es:
                    matched = (es, ee, suf)
                    break
            if matched is not None:
                if prev_entity is None or matched != prev_entity:
                    label = f"B-{matched[2]}"
                else:
                    label = f"I-{matched[2]}"
                prev_entity = matched
            else:
                prev_entity = None

            ner_ids[i] = self.ner_label_to_id.get(label, 0)

        intent_labels = torch.tensor(_intent_id(sample.get("intent")), dtype=torch.long)
        action_labels = torch.tensor(_action_id(sample.get("actions")), dtype=torch.long)

        sentiment = sample.get("sentiment")
        try:
            sentiment_f = float(sentiment) if sentiment is not None else 0.0
        except (TypeError, ValueError):
            sentiment_f = 0.0
        sentiment_labels = torch.tensor(sentiment_f, dtype=torch.float32)

        weight = sample.get("weight")
        try:
            weight_f = float(weight) if weight is not None else 1.0
        except (TypeError, ValueError):
            weight_f = 1.0
        weight_t = torch.tensor(weight_f, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "ner_labels": torch.tensor(ner_ids, dtype=torch.long),
            "intent_labels": intent_labels,
            "action_labels": action_labels,
            "sentiment_labels": sentiment_labels,
            "weight": weight_t,
        }


def build_dataloaders(
    *,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    tokenizer_name: str,
    batch_size: int = 32,
    num_workers: int = 0,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = DatasetConfig(tokenizer_name=tokenizer_name, max_length=max_length)
    train_ds = ConversationDataset(_read_samples(train_path), cfg)
    val_ds = ConversationDataset(_read_samples(val_path), cfg)
    test_ds = ConversationDataset(_read_samples(test_path), cfg)

    def _loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, int(num_workers)),
            pin_memory=torch.cuda.is_available(),
        )

    return _loader(train_ds, True), _loader(val_ds, False), _loader(test_ds, False)

