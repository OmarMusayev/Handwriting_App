# app/core/transformer_singleton.py
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class TransformerSingleton:
    _ready = False
    model = None
    text_vocab = None
    stroke_vocab = None
    stroke_tokenizer = None
    max_gen_tokens = 1024
    device = None
    _args = None

    @classmethod
    def initialize(cls, checkpoint_path: str, device: str = "cpu"):
        if cls._ready:
            return

        from handwriting.checkpoint import load_checkpoint_payload, restore_model_weights
        from handwriting.model import build_model
        from handwriting.data import TextVocab
        from handwriting.tokenizers import (
            PolarOffsetTokenizer,
            StrokeVocab,
            TokenizerSpec,
            RadiusCodebook,
            RadiusCodebooks,
        )

        path = Path(checkpoint_path)
        if not path.exists():
            print(f"[transformer] checkpoint not found: {path} — transformer mode disabled")
            return

        payload = load_checkpoint_payload(path)
        checkpoint_args = SimpleNamespace(**payload["args"])

        # Reconstruct text vocab from checkpoint
        text_vocab = TextVocab([])
        text_vocab.itos = [str(t) for t in payload["text_vocab_itos"]]
        text_vocab.stoi = {t: i for i, t in enumerate(text_vocab.itos)}
        text_vocab.pad_id = text_vocab.stoi[TextVocab.PAD]
        text_vocab.bos_id = text_vocab.stoi[TextVocab.BOS]
        text_vocab.eos_id = text_vocab.stoi[TextVocab.EOS]
        text_vocab.unk_id = text_vocab.stoi[TextVocab.UNK]

        # Reconstruct tokenizer spec
        ts = payload["tokenizer_spec"]
        tokenizer_spec = TokenizerSpec(
            name=str(ts["name"]),
            scheme=str(ts["scheme"]),
            n_angle_bins=int(ts["n_angle_bins"]),
            n_radius_bins=int(ts["n_radius_bins"]),
            max_points_for_bins=int(ts.get("max_points_for_bins", 500_000)),
            radius_codebook_mode=str(ts.get("radius_codebook_mode", "shared")),
            radius_decode_mode=str(ts.get("radius_decode_mode", "midpoint")),
            n_radius_bins_draw=int(ts["n_radius_bins_draw"]) if ts.get("n_radius_bins_draw") is not None else None,
            n_radius_bins_nondraw=int(ts["n_radius_bins_nondraw"]) if ts.get("n_radius_bins_nondraw") is not None else None,
        )

        # Reconstruct radius codebooks
        rc = payload["radius_codebooks"]

        def _codebook(d):
            if d is None:
                return None
            return RadiusCodebook(
                edges=np.asarray(d["edges"], dtype=np.float32),
                midpoints=np.asarray(d["midpoints"], dtype=np.float32),
                medians=np.asarray(d["medians"], dtype=np.float32),
                decode_values=np.asarray(d["decode_values"], dtype=np.float32),
                sample_count=int(d.get("sample_count", 0)),
            )

        radius_codebooks = RadiusCodebooks(
            mode=str(rc["mode"]),
            decode_mode=str(rc["decode_mode"]),
            shared=_codebook(rc.get("shared")),
            nondraw=_codebook(rc.get("draw_flag_0")),
            draw=_codebook(rc.get("draw_flag_1")),
        )

        stroke_vocab = StrokeVocab(tokenizer_spec)
        stroke_tokenizer = PolarOffsetTokenizer(tokenizer_spec, radius_codebooks, stroke_vocab)

        # Build and load model
        dev = torch.device(device)
        model = build_model(checkpoint_args, text_vocab, stroke_vocab).to(dev)
        restore_model_weights(model, payload)
        model.eval()

        cls.model = model
        cls.text_vocab = text_vocab
        cls.stroke_vocab = stroke_vocab
        cls.stroke_tokenizer = stroke_tokenizer
        cls.device = dev
        cls._args = checkpoint_args
        cls.max_gen_tokens = int(
            getattr(checkpoint_args, "sample_max_gen_tokens",
                    getattr(checkpoint_args, "max_stroke_tokens", 1024))
        )
        cls._ready = True
        print(f"[transformer] loaded from {path}")

    @classmethod
    def is_available(cls) -> bool:
        return cls._ready

    @classmethod
    def generate_sample(cls, text: str, seed: int = None) -> np.ndarray:
        """Generate one handwriting sample. Returns absolute points (N, 3)."""
        if not cls._ready:
            raise RuntimeError("Transformer model not loaded")

        from handwriting.generation import generate_tokens, offsets_to_absolute_points

        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        token_ids = generate_tokens(
            model=cls.model,
            text=text,
            text_vocab=cls.text_vocab,
            stroke_vocab=cls.stroke_vocab,
            max_text_len=int(getattr(cls._args, "max_text_len", 64)),
            max_gen_tokens=cls.max_gen_tokens,
            device=cls.device,
            temperature=0.9,
            top_k=20,
            greedy=False,
            sample_seed=seed,
        )

        offsets = cls.stroke_tokenizer.decode_tokens_to_offsets(token_ids)
        points = offsets_to_absolute_points(offsets)
        return points
