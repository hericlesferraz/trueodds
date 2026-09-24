"""The cross-encoder (D2): one score per (state, question, option), softmaxed across options.

A batch holds B questions with up to K_max options each, flattened to the N real sequences only:
padded options are never encoded. Their score is -inf, so the softmax gives them exactly 0.

The shared-state variant (D13, spec 004, `trueodds.shared`) reads packed sequences instead: the
state once, then each question and its options, with masks that keep them apart. Each option is
scored from its own [CLS] in the packed sequence; the head and the -inf padding are the same.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

POOLINGS = ("cls", "mean")
ARCHITECTURES = ("cross", "shared")


class DecisionModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        pooling: str = "cls",
        backbone: str | None = None,
        architecture: str = "cross",
    ):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, not {pooling!r}")
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}, not {architecture!r}")
        if architecture == "shared" and pooling != "cls":
            raise ValueError("the shared-state model pools each option's [CLS] (spec 004)")
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_size, 1)
        self.pooling = pooling
        self.backbone = backbone
        self.architecture = architecture

    @classmethod
    def from_pretrained(
        cls, backbone: str, pooling: str = "cls", attn: str = "sdpa", architecture: str = "cross"
    ):
        encoder = AutoModel.from_pretrained(backbone, attn_implementation=attn)
        return cls(encoder, pooling, backbone, architecture)

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        pooling: str = "cls",
        attn: str = "sdpa",
        architecture: str = "cross",
    ):
        """A randomly initialized model; the unit tests use a tiny one."""
        return cls(
            AutoModel.from_config(config, attn_implementation=attn), pooling, None, architecture
        )

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[N, L] token ids -> [N] scores, one per sequence."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        return self.head(pooled).squeeze(-1)

    def score_packed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        question_ids: torch.Tensor,
        option_ids: torch.Tensor,
        pool_row: torch.Tensor,
        pool_col: torch.Tensor,
    ) -> torch.Tensor:
        """[R, L] packed sequences -> [N] scores, one per option, read at its [CLS]."""
        from trueodds.shared import attention_masks

        masks = attention_masks(
            position_ids,
            question_ids,
            option_ids,
            attention_mask,
            self.encoder.config.sliding_window,
        )
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=masks, position_ids=position_ids
        ).last_hidden_state
        return self.head(hidden[pool_row, pool_col]).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_index: torch.Tensor,
        option_index: torch.Tensor,
        n_questions: int,
        k_max: int,
        **packed: torch.Tensor,
    ) -> torch.Tensor:
        """[B, K_max] scores in float32; -inf where a question has fewer than K_max options.

        `packed` holds the shared-state inputs (`PackedBatch.model_inputs`); without them the
        batch is the cross-encoder's, one sequence per option.
        """
        if packed:
            s = self.score_packed(input_ids, attention_mask, **packed).float()
        else:
            s = self.score(input_ids, attention_mask).float()
        scores = s.new_full((n_questions, k_max), float("-inf"))
        return scores.index_put((question_index, option_index), s)

    def save(self, path: Path, tokenizer=None, max_len: int | None = None) -> None:
        """`model.safetensors`, the encoder's `config.json`, `model.json`, and the tokenizer."""
        path.mkdir(parents=True, exist_ok=True)
        save_model(self, str(path / "model.safetensors"))
        self.encoder.config.save_pretrained(path)
        meta = {
            "backbone": self.backbone,
            "pooling": self.pooling,
            "max_len": max_len,
            "architecture": self.architecture,
        }
        (path / "model.json").write_text(json.dumps(meta, indent=2) + "\n")
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: Path, attn: str = "sdpa", device: str | torch.device = "cpu"):
        meta = json.loads((path / "model.json").read_text())
        config = AutoConfig.from_pretrained(path)
        encoder = AutoModel.from_config(config, attn_implementation=attn)
        # Checkpoints written before Phase 5 have no architecture: they are cross-encoders.
        model = cls(encoder, meta["pooling"], meta["backbone"], meta.get("architecture", "cross"))
        load_model(model, str(path / "model.safetensors"), device=str(device))
        return model.to(device)


def read_meta(path: Path) -> dict:
    return json.loads((path / "model.json").read_text())
