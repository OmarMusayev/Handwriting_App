from __future__ import annotations

import torch
import torch.nn as nn

from .data import TextVocab
from .tokenizers import StrokeVocab


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        causal_mask: torch.Tensor,
        x_pad_mask: torch.Tensor | None,
        memory_pad_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=x_pad_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        h = self.norm2(x)
        h, _ = self.cross_attn(
            h,
            memory,
            memory,
            key_padding_mask=memory_pad_mask,
            need_weights=False,
        )
        x = x + self.dropout(h)

        h = self.norm3(x)
        x = x + self.mlp(h)
        return x


class CrossAttentionGPT(nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        stroke_vocab_size: int,
        max_text_len: int,
        max_stroke_len: int,
        num_writers: int = 1,
        use_writer_conditioning: bool = False,
        writer_embed_dim: int = 64,
        writer_conditioning_mode: str = "add_to_both",
        num_style_clusters: int = 1,
        use_style_cluster_conditioning: bool = False,
        style_cluster_embed_dim: int = 48,
        cluster_conditioning_mode: str = "add_to_both",
        d_model: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_text_len = int(max_text_len)
        self.max_stroke_len = int(max_stroke_len)
        self.use_writer_conditioning = bool(use_writer_conditioning)
        self.writer_conditioning_mode = str(writer_conditioning_mode)
        self.use_style_cluster_conditioning = bool(use_style_cluster_conditioning)
        self.cluster_conditioning_mode = str(cluster_conditioning_mode)

        self.text_embed = nn.Embedding(text_vocab_size, d_model)
        self.stroke_embed = nn.Embedding(stroke_vocab_size, d_model)
        self.text_pos = nn.Embedding(max_text_len, d_model)
        self.stroke_pos = nn.Embedding(max_stroke_len, d_model)
        if self.use_writer_conditioning:
            self.writer_embed = nn.Embedding(int(num_writers), int(writer_embed_dim))
            self.writer_proj = (
                nn.Identity()
                if int(writer_embed_dim) == int(d_model)
                else nn.Linear(int(writer_embed_dim), int(d_model))
            )
        else:
            self.writer_embed = None
            self.writer_proj = None
        if self.use_style_cluster_conditioning:
            self.style_cluster_embed = nn.Embedding(int(num_style_clusters), int(style_cluster_embed_dim))
            self.style_cluster_proj = (
                nn.Identity()
                if int(style_cluster_embed_dim) == int(d_model)
                else nn.Linear(int(style_cluster_embed_dim), int(d_model))
            )
        else:
            self.style_cluster_embed = None
            self.style_cluster_proj = None

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, stroke_vocab_size, bias=False)

    def forward(
        self,
        text_ids: torch.Tensor,
        stroke_in_ids: torch.Tensor,
        text_pad_mask: torch.Tensor | None = None,
        stroke_pad_mask: torch.Tensor | None = None,
        writer_ids: torch.Tensor | None = None,
        style_cluster_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, text_len = text_ids.shape
        batch_size_2, stroke_len = stroke_in_ids.shape
        assert batch_size == batch_size_2
        if text_len > self.max_text_len:
            raise ValueError(f"Text length {text_len} exceeds max_text_len {self.max_text_len}")
        if stroke_len > self.max_stroke_len:
            raise ValueError(f"Stroke length {stroke_len} exceeds max_stroke_len {self.max_stroke_len}")
        if self.use_writer_conditioning and writer_ids is None:
            raise ValueError("writer_ids must be provided when use_writer_conditioning=True")
        if self.use_style_cluster_conditioning and style_cluster_ids is None:
            raise ValueError("style_cluster_ids must be provided when use_style_cluster_conditioning=True")

        text_pos_ids = torch.arange(text_len, device=text_ids.device).unsqueeze(0)
        stroke_pos_ids = torch.arange(stroke_len, device=stroke_in_ids.device).unsqueeze(0)

        memory = self.text_embed(text_ids) + self.text_pos(text_pos_ids)
        x = self.stroke_embed(stroke_in_ids) + self.stroke_pos(stroke_pos_ids)
        if self.use_writer_conditioning:
            writer_context = self.writer_proj(self.writer_embed(writer_ids)).unsqueeze(1)
            if self.writer_conditioning_mode in {"add_to_text", "add_to_both"}:
                memory = memory + writer_context
            if self.writer_conditioning_mode in {"add_to_stroke", "add_to_both"}:
                x = x + writer_context
        if self.use_style_cluster_conditioning:
            cluster_context = self.style_cluster_proj(self.style_cluster_embed(style_cluster_ids)).unsqueeze(1)
            if self.cluster_conditioning_mode in {"add_to_text", "add_to_both"}:
                memory = memory + cluster_context
            if self.cluster_conditioning_mode in {"add_to_stroke", "add_to_both"}:
                x = x + cluster_context

        memory = self.drop(memory)
        x = self.drop(x)

        causal_mask = torch.triu(
            torch.ones(stroke_len, stroke_len, device=stroke_in_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, memory, causal_mask, stroke_pad_mask, text_pad_mask)

        x = self.final_norm(x)
        return self.lm_head(x)


def build_model(args, text_vocab: TextVocab, stroke_vocab: StrokeVocab) -> CrossAttentionGPT:
    return CrossAttentionGPT(
        text_vocab_size=len(text_vocab),
        stroke_vocab_size=stroke_vocab.vocab_size,
        max_text_len=args.max_text_len,
        max_stroke_len=args.max_stroke_tokens,
        num_writers=int(getattr(args, "num_writer_embeddings", 1)),
        use_writer_conditioning=bool(getattr(args, "use_writer_conditioning", False)),
        writer_embed_dim=int(getattr(args, "writer_embed_dim", args.d_model)),
        writer_conditioning_mode=str(getattr(args, "writer_conditioning_mode", "add_to_both")),
        num_style_clusters=int(getattr(args, "num_style_clusters", 1)),
        use_style_cluster_conditioning=bool(getattr(args, "use_style_cluster_conditioning", False)),
        style_cluster_embed_dim=int(getattr(args, "style_cluster_embed_dim", args.d_model)),
        cluster_conditioning_mode=str(getattr(args, "cluster_conditioning_mode", "add_to_both")),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
