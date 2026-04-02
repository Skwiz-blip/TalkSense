"""
conversation_analyzer.py
Multi-task BERT model for conversation analysis.

Tasks:
  1. Named Entity Recognition  (NER)   — BIO tagging
  2. Intent Classification             — 5 classes
  3. Action Extraction                 — 5 classes
  4. Sentiment Analysis                — regression [-1, 1]
  5. Relation Extraction               — 5 relation types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel

logger = logging.getLogger(__name__)


# ── Label sets ──────────────────────────────────────────────
NER_LABELS = [
    "O",
    "B-PERSON", "I-PERSON",
    "B-DATE",   "I-DATE",
    "B-PROJECT","I-PROJECT",
    "B-AMOUNT", "I-AMOUNT",
    "B-ORG",    "I-ORG",
]

INTENT_LABELS = [
    "information_request",
    "decision_making",
    "problem_solving",
    "status_update",
    "planning",
]

ACTION_LABELS = [
    "none",
    "create_task",
    "create_reminder",
    "follow_up",
    "create_alert",
]

RELATION_LABELS = [
    "none",
    "assigned_to",
    "blocks",
    "depends_on",
    "relates_to",
    "caused_by",
]

NUM_NER_LABELS      = len(NER_LABELS)       # 11
NUM_INTENT_LABELS   = len(INTENT_LABELS)    # 5
NUM_ACTION_LABELS   = len(ACTION_LABELS)    # 5
NUM_RELATION_LABELS = len(RELATION_LABELS)  # 6


# ── Config extension ─────────────────────────────────────────
@dataclass
class AnalyzerConfig:
    """Extended config stored alongside BertConfig."""
    ner_dropout:      float = 0.10
    intent_dropout:   float = 0.15
    action_dropout:   float = 0.10
    relation_dropout: float = 0.10
    hidden_mid:       int   = 512
    action_mid:       int   = 256


# ── Model ────────────────────────────────────────────────────
class ConversationAnalyzerModel(BertPreTrainedModel):
    """
    BERT-based multi-task model. All 5 tasks share the BERT backbone;
    each task has its own classification head.
    """

    def __init__(self, config: BertConfig, analyzer_cfg: Optional[AnalyzerConfig] = None):
        super().__init__(config)
        self.cfg = analyzer_cfg or AnalyzerConfig()

        # ── Shared backbone ──────────────────────────
        self.bert = BertModel(config)

        H = config.hidden_size  # 768 for bert-base

        # ── Task 1: NER (token-level) ────────────────
        self.ner_head = nn.Sequential(
            nn.Linear(H, self.cfg.hidden_mid),
            nn.GELU(),
            nn.Dropout(self.cfg.ner_dropout),
            nn.Linear(self.cfg.hidden_mid, NUM_NER_LABELS),
        )

        # ── Task 2: Intent (CLS-level) ───────────────
        self.intent_head = nn.Sequential(
            nn.Dropout(self.cfg.intent_dropout),
            nn.Linear(H, NUM_INTENT_LABELS),
        )

        # ── Task 3: Action (CLS-level) ───────────────
        self.action_head = nn.Sequential(
            nn.Linear(H, self.cfg.action_mid),
            nn.GELU(),
            nn.Dropout(self.cfg.action_dropout),
            nn.Linear(self.cfg.action_mid, NUM_ACTION_LABELS),
        )

        # ── Task 4: Sentiment (CLS-level, regression) ─
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(H, 1),
            nn.Tanh(),          # output ∈ [-1, 1]
        )

        # ── Task 5: Relation (pair-level) ────────────
        # Input = concat of two entity representations (2H)
        self.relation_head = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.GELU(),
            nn.Dropout(self.cfg.relation_dropout),
            nn.Linear(H, NUM_RELATION_LABELS),
        )

        # ── CRF for NER (optional, improves tagging) ──
        # Replaced by vanilla cross-entropy for simplicity in this version.
        # To enable CRF: pip install torchcrf and swap the NER loss in Trainer.

        self.init_weights()
        logger.info(
            "ConversationAnalyzerModel initialized | "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(
        self,
        input_ids:      torch.Tensor,               # [B, L]
        attention_mask: torch.Tensor,               # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,
        # Ground-truth labels (only during training)
        ner_labels:     Optional[torch.Tensor] = None,  # [B, L]
        intent_labels:  Optional[torch.Tensor] = None,  # [B]
        action_labels:  Optional[torch.Tensor] = None,  # [B]
        sentiment_labels: Optional[torch.Tensor] = None,# [B]
        entity_pairs:   Optional[torch.Tensor] = None,  # [B, P, 2]
        relation_labels:Optional[torch.Tensor] = None,  # [B, P]
    ) -> Dict[str, torch.Tensor]:

        # ── BERT forward ──────────────────────────────
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        seq_out  = outputs.last_hidden_state   # [B, L, H]
        pool_out = outputs.pooler_output       # [B, H]

        # ── Task outputs ──────────────────────────────
        ner_logits      = self.ner_head(seq_out)        # [B, L, num_ner]
        intent_logits   = self.intent_head(pool_out)    # [B, num_intent]
        action_logits   = self.action_head(pool_out)    # [B, num_action]
        sentiment_logits= self.sentiment_head(pool_out) # [B, 1]

        result: Dict[str, torch.Tensor] = {
            "ner_logits":       ner_logits,
            "intent_logits":    intent_logits,
            "action_logits":    action_logits,
            "sentiment_logits": sentiment_logits,
        }

        # ── Relation extraction (only when pairs provided) ──
        if entity_pairs is not None:
            # entity_pairs: [B, P, 2] — start token indices of each entity pair
            relation_logits = self._compute_relations(seq_out, entity_pairs)
            result["relation_logits"] = relation_logits   # [B, P, num_rel]

        # ── Compute loss when labels present ──────────
        if ner_labels is not None or intent_labels is not None:
            loss = self._compute_loss(
                ner_logits, intent_logits, action_logits, sentiment_logits,
                ner_labels, intent_labels, action_labels, sentiment_labels,
                result.get("relation_logits"), relation_labels,
                attention_mask,
            )
            result["loss"] = loss

        return result

    def _compute_relations(
        self,
        seq_out:      torch.Tensor,   # [B, L, H]
        entity_pairs: torch.Tensor,   # [B, P, 2]
    ) -> torch.Tensor:
        B, P, _ = entity_pairs.shape
        H       = seq_out.size(-1)
        pairs   = []
        for b in range(B):
            for p in range(P):
                i1, i2 = entity_pairs[b, p, 0].item(), entity_pairs[b, p, 1].item()
                e1 = seq_out[b, i1]  # [H]
                e2 = seq_out[b, i2]  # [H]
                pairs.append(torch.cat([e1, e2]))  # [2H]
        if not pairs:
            return torch.zeros(B, P, NUM_RELATION_LABELS, device=seq_out.device)
        stacked = torch.stack(pairs).view(B, P, H * 2)
        return self.relation_head(stacked)  # [B, P, num_rel]

    # ── Multi-task loss ───────────────────────────────
    TASK_WEIGHTS = {
        "ner":       0.30,
        "intent":    0.20,
        "action":    0.25,
        "sentiment": 0.10,
        "relation":  0.15,
    }

    def _compute_loss(
        self,
        ner_logits, intent_logits, action_logits, sentiment_logits,
        ner_labels, intent_labels, action_labels, sentiment_labels,
        relation_logits, relation_labels,
        attention_mask,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=ner_logits.device)
        w     = self.TASK_WEIGHTS

        # NER — cross-entropy, ignore padding (-100)
        if ner_labels is not None:
            B, L, C = ner_logits.shape
            active   = attention_mask.view(-1) == 1
            flat_log = ner_logits.view(-1, C)[active]
            flat_lbl = ner_labels.view(-1)[active]
            total    = total + w["ner"] * F.cross_entropy(flat_log, flat_lbl, ignore_index=-100)

        # Intent — cross-entropy
        if intent_labels is not None:
            total = total + w["intent"] * F.cross_entropy(intent_logits, intent_labels)

        # Action — cross-entropy
        if action_labels is not None:
            total = total + w["action"] * F.cross_entropy(action_logits, action_labels)

        # Sentiment — MSE
        if sentiment_labels is not None:
            total = total + w["sentiment"] * F.mse_loss(
                sentiment_logits.squeeze(-1),
                sentiment_labels.float(),
            )

        # Relation — cross-entropy
        if relation_logits is not None and relation_labels is not None:
            B, P, C = relation_logits.shape
            total   = total + w["relation"] * F.cross_entropy(
                relation_logits.view(-1, C),
                relation_labels.view(-1),
                ignore_index=-100,
            )

        return total

    # ── Convenience decoders (inference) ─────────────
    @torch.no_grad()
    def predict_ner(
        self, logits: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[Tuple[str, float]]]:
        """Returns per-batch list of (label, confidence) per token."""
        probs  = F.softmax(logits, dim=-1)          # [B, L, C]
        preds  = probs.argmax(dim=-1)               # [B, L]
        confs  = probs.max(dim=-1).values           # [B, L]
        result = []
        for b in range(logits.size(0)):
            seq = []
            for t in range(logits.size(1)):
                if attention_mask[b, t] == 0:
                    break
                seq.append((NER_LABELS[preds[b, t].item()], confs[b, t].item()))
            result.append(seq)
        return result

    @torch.no_grad()
    def predict_intent(self, logits: torch.Tensor) -> List[Tuple[str, float]]:
        probs  = F.softmax(logits, dim=-1)
        preds  = probs.argmax(dim=-1)
        confs  = probs.max(dim=-1).values
        return [
            (INTENT_LABELS[preds[b].item()], confs[b].item())
            for b in range(logits.size(0))
        ]

    @torch.no_grad()
    def predict_actions(self, logits: torch.Tensor) -> List[List[Tuple[str, float]]]:
        probs = F.softmax(logits, dim=-1)
        result = []
        for b in range(logits.size(0)):
            row = [
                (ACTION_LABELS[i], probs[b, i].item())
                for i in range(NUM_ACTION_LABELS)
                if i > 0 and probs[b, i].item() > 0.4
            ]
            result.append(row)
        return result

    @torch.no_grad()
    def predict_sentiment(self, logits: torch.Tensor) -> List[float]:
        return logits.squeeze(-1).tolist()


# ── Factory ───────────────────────────────────────────────────
def build_model(
    pretrained_name: str = "bert-base-multilingual-cased",
    analyzer_cfg:    Optional[AnalyzerConfig] = None,
    *,
    load_weights: bool = True,
) -> ConversationAnalyzerModel:
    """Instantiate from a HuggingFace BERT checkpoint."""
    config = BertConfig.from_pretrained(pretrained_name)
    # Add task-specific dimensions to config (serialized with model)
    config.num_labels_ner      = NUM_NER_LABELS
    config.num_labels_intent   = NUM_INTENT_LABELS
    config.num_labels_action   = NUM_ACTION_LABELS
    config.num_labels_relation = NUM_RELATION_LABELS

    if load_weights:
        return ConversationAnalyzerModel.from_pretrained(
            pretrained_name,
            config       = config,
            analyzer_cfg = analyzer_cfg,
            ignore_mismatched_sizes = True,
        )

    # Config-only init: avoids downloading large model weights (useful for low disk / offline).
    return ConversationAnalyzerModel(config=config, analyzer_cfg=analyzer_cfg)
