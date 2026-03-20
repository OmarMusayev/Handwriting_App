from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    scheme: str
    n_angle_bins: int
    n_radius_bins: int
    max_points_for_bins: int = 500_000
    radius_codebook_mode: str = "shared"
    radius_decode_mode: str = "midpoint"
    n_radius_bins_draw: Optional[int] = None
    n_radius_bins_nondraw: Optional[int] = None

    @property
    def tokens_per_offset(self) -> int:
        if self.scheme == "polar_2token":
            return 2
        if self.scheme == "polar_3token":
            return 3
        raise ValueError(f"Unsupported tokenizer scheme: {self.scheme}")

    @property
    def draw_radius_bins(self) -> int:
        return int(self.n_radius_bins if self.n_radius_bins_draw is None else self.n_radius_bins_draw)

    @property
    def nondraw_radius_bins(self) -> int:
        return int(self.n_radius_bins if self.n_radius_bins_nondraw is None else self.n_radius_bins_nondraw)

    @property
    def max_radius_bins(self) -> int:
        return max(self.draw_radius_bins, self.nondraw_radius_bins)

    def radius_bins_for_flag(self, draw_flag: int) -> int:
        return self.draw_radius_bins if int(draw_flag) == 1 else self.nondraw_radius_bins


BUILTIN_TOKENIZER_SPECS: Dict[str, TokenizerSpec] = {
    "baseline_2token_64x32": TokenizerSpec(
        name="baseline_2token_64x32",
        scheme="polar_2token",
        n_angle_bins=64,
        n_radius_bins=32,
    ),
    "finer_2token_128x64": TokenizerSpec(
        name="finer_2token_128x64",
        scheme="polar_2token",
        n_angle_bins=128,
        n_radius_bins=64,
    ),
    "new_3token_64x32": TokenizerSpec(
        name="new_3token_64x32",
        scheme="polar_3token",
        n_angle_bins=64,
        n_radius_bins=32,
    ),
    "finer_2token_128x64_median_decode": TokenizerSpec(
        name="finer_2token_128x64_median_decode",
        scheme="polar_2token",
        n_angle_bins=128,
        n_radius_bins=64,
        radius_codebook_mode="shared",
        radius_decode_mode="empirical_median",
    ),
    "finer_2token_128x64_split_radius": TokenizerSpec(
        name="finer_2token_128x64_split_radius",
        scheme="polar_2token",
        n_angle_bins=128,
        n_radius_bins=64,
        radius_codebook_mode="by_draw_flag",
        radius_decode_mode="midpoint",
    ),
    "finer_2token_128x64_split_radius_median_decode": TokenizerSpec(
        name="finer_2token_128x64_split_radius_median_decode",
        scheme="polar_2token",
        n_angle_bins=128,
        n_radius_bins=64,
        radius_codebook_mode="by_draw_flag",
        radius_decode_mode="empirical_median",
    ),
    "finer_2token_128x64_split_radius_median_decode_nondraw128": TokenizerSpec(
        name="finer_2token_128x64_split_radius_median_decode_nondraw128",
        scheme="polar_2token",
        n_angle_bins=128,
        n_radius_bins=64,
        radius_codebook_mode="by_draw_flag",
        radius_decode_mode="empirical_median",
        n_radius_bins_draw=64,
        n_radius_bins_nondraw=128,
    ),
}


ALLOWED_TOKENIZER_SCHEMES = {"polar_2token", "polar_3token"}
ALLOWED_RADIUS_CODEBOOK_MODES = {"shared", "by_draw_flag"}
ALLOWED_RADIUS_DECODE_MODES = {"midpoint", "empirical_median"}


@dataclass
class RadiusCodebook:
    edges: np.ndarray
    midpoints: np.ndarray
    medians: np.ndarray
    decode_values: np.ndarray
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "sample_count": int(self.sample_count),
            "edges": self.edges.tolist(),
            "midpoints": self.midpoints.tolist(),
            "medians": self.medians.tolist(),
            "decode_values": self.decode_values.tolist(),
        }


@dataclass
class RadiusCodebooks:
    mode: str
    decode_mode: str
    shared: Optional[RadiusCodebook] = None
    nondraw: Optional[RadiusCodebook] = None
    draw: Optional[RadiusCodebook] = None

    def codebook_for_flag(self, draw_flag: int) -> RadiusCodebook:
        if self.mode == "shared":
            if self.shared is None:
                raise ValueError("Shared radius codebook is missing")
            return self.shared
        if int(draw_flag) == 1:
            if self.draw is None:
                raise ValueError("Draw radius codebook is missing")
            return self.draw
        if self.nondraw is None:
            raise ValueError("Non-draw radius codebook is missing")
        return self.nondraw

    def legacy_radius_edges(self) -> np.ndarray:
        if self.shared is not None:
            return self.shared.edges
        if self.draw is not None:
            return self.draw.edges
        if self.nondraw is not None:
            return self.nondraw.edges
        raise ValueError("No radius codebook edges are available")

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "decode_mode": self.decode_mode,
            "shared": self.shared.to_dict() if self.shared is not None else None,
            "draw_flag_0": self.nondraw.to_dict() if self.nondraw is not None else None,
            "draw_flag_1": self.draw.to_dict() if self.draw is not None else None,
        }


def tokenizer_spec_to_dict(spec: TokenizerSpec) -> dict:
    return {
        "name": spec.name,
        "scheme": spec.scheme,
        "n_angle_bins": spec.n_angle_bins,
        "n_radius_bins": spec.n_radius_bins,
        "n_radius_bins_draw": spec.draw_radius_bins,
        "n_radius_bins_nondraw": spec.nondraw_radius_bins,
        "max_points_for_bins": spec.max_points_for_bins,
        "radius_codebook_mode": spec.radius_codebook_mode,
        "radius_decode_mode": spec.radius_decode_mode,
    }


def _validate_tokenizer_spec(spec: TokenizerSpec) -> TokenizerSpec:
    if spec.scheme not in ALLOWED_TOKENIZER_SCHEMES:
        raise ValueError(f"Unsupported tokenizer scheme: {spec.scheme}")
    if spec.radius_codebook_mode not in ALLOWED_RADIUS_CODEBOOK_MODES:
        raise ValueError(f"Unsupported radius codebook mode: {spec.radius_codebook_mode}")
    if spec.radius_decode_mode not in ALLOWED_RADIUS_DECODE_MODES:
        raise ValueError(f"Unsupported radius decode mode: {spec.radius_decode_mode}")
    if spec.n_angle_bins <= 0:
        raise ValueError("n_angle_bins must be positive")
    if spec.draw_radius_bins <= 0 or spec.nondraw_radius_bins <= 0:
        raise ValueError("radius bin counts must be positive")
    if spec.scheme == "polar_3token" and spec.draw_radius_bins != spec.nondraw_radius_bins:
        raise ValueError("polar_3token requires matching draw and non-draw radius bin counts")
    if spec.radius_codebook_mode == "shared" and spec.draw_radius_bins != spec.nondraw_radius_bins:
        raise ValueError("Shared radius codebooks require matching draw and non-draw radius bin counts")
    return spec


def _collect_radii_from_offsets(
    offsets_array: Sequence[np.ndarray],
    *,
    max_points_for_bins: int,
    seed: int,
    draw_flag: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed + (0 if draw_flag is None else int(draw_flag) + 1))
    radii_chunks: List[np.ndarray] = []
    collected = 0
    for offsets in offsets_array:
        arr = np.asarray(offsets, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3 or arr.shape[0] == 0:
            continue
        if draw_flag is not None:
            mask = np.clip(np.rint(arr[:, 2]), 0, 1).astype(np.int64) == int(draw_flag)
            arr = arr[mask]
            if arr.shape[0] == 0:
                continue
        radii = np.sqrt(arr[:, 0] * arr[:, 0] + arr[:, 1] * arr[:, 1])
        radii = radii[np.isfinite(radii)]
        if radii.size == 0:
            continue
        remaining = max_points_for_bins - collected
        if remaining <= 0:
            break
        if radii.size > remaining:
            indices = rng.choice(radii.size, size=remaining, replace=False)
            radii = radii[indices]
        radii_chunks.append(radii)
        collected += radii.size
        if collected >= max_points_for_bins:
            break

    if not radii_chunks:
        return np.zeros((0,), dtype=np.float32)

    radii = np.concatenate(radii_chunks)
    radii = radii[np.isfinite(radii)]
    radii = radii[radii >= 0]
    return np.asarray(radii, dtype=np.float32)


def _build_radius_edges(radii: np.ndarray, n_radius_bins: int) -> np.ndarray:
    if radii.size == 0:
        raise RuntimeError("No valid radii after filtering")
    quantiles = np.linspace(0.0, 1.0, n_radius_bins + 1)
    edges = np.quantile(radii, quantiles).astype(np.float32)
    if np.allclose(edges[0], edges[-1]):
        hi = float(edges[0]) + 1e-3
        edges = np.linspace(float(edges[0]), hi, n_radius_bins + 1, dtype=np.float32)
    else:
        for index in range(1, len(edges)):
            if edges[index] <= edges[index - 1]:
                edges[index] = np.nextafter(edges[index - 1], np.float32(np.inf))
    return edges


def _build_radius_codebook(radii: np.ndarray, n_radius_bins: int, decode_mode: str) -> RadiusCodebook:
    edges = _build_radius_edges(radii, n_radius_bins=n_radius_bins)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    medians = midpoints.copy()
    radius_bins = np.searchsorted(edges, radii, side="right") - 1
    radius_bins = np.clip(radius_bins, 0, n_radius_bins - 1)
    for radius_bin in range(n_radius_bins):
        mask = radius_bins == radius_bin
        if np.any(mask):
            medians[radius_bin] = float(np.median(radii[mask]))
    decode_values = medians if decode_mode == "empirical_median" else midpoints
    return RadiusCodebook(
        edges=np.asarray(edges, dtype=np.float32),
        midpoints=np.asarray(midpoints, dtype=np.float32),
        medians=np.asarray(medians, dtype=np.float32),
        decode_values=np.asarray(decode_values, dtype=np.float32),
        sample_count=int(radii.size),
    )


def collect_radius_codebooks_from_offsets(
    offsets_array: Sequence[np.ndarray],
    spec: TokenizerSpec,
    max_points_for_bins: int,
    seed: int,
) -> RadiusCodebooks:
    shared_radii = _collect_radii_from_offsets(
        offsets_array,
        max_points_for_bins=max_points_for_bins,
        seed=seed,
        draw_flag=None,
    )
    if shared_radii.size == 0:
        raise RuntimeError("Could not collect radii from train offsets")

    if spec.radius_codebook_mode == "shared":
        return RadiusCodebooks(
            mode=spec.radius_codebook_mode,
            decode_mode=spec.radius_decode_mode,
            shared=_build_radius_codebook(
                shared_radii,
                spec.radius_bins_for_flag(1),
                spec.radius_decode_mode,
            ),
        )

    nondraw_radii = _collect_radii_from_offsets(
        offsets_array,
        max_points_for_bins=max_points_for_bins,
        seed=seed + 17,
        draw_flag=0,
    )
    draw_radii = _collect_radii_from_offsets(
        offsets_array,
        max_points_for_bins=max_points_for_bins,
        seed=seed + 31,
        draw_flag=1,
    )
    if nondraw_radii.size == 0:
        nondraw_radii = shared_radii
    if draw_radii.size == 0:
        draw_radii = shared_radii
    return RadiusCodebooks(
        mode=spec.radius_codebook_mode,
        decode_mode=spec.radius_decode_mode,
        nondraw=_build_radius_codebook(
            nondraw_radii,
            spec.radius_bins_for_flag(0),
            spec.radius_decode_mode,
        ),
        draw=_build_radius_codebook(
            draw_radii,
            spec.radius_bins_for_flag(1),
            spec.radius_decode_mode,
        ),
    )


class StrokeVocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, spec: TokenizerSpec):
        self.spec = spec
        self.n_angle_bins = int(spec.n_angle_bins)
        self.n_radius_bins = int(spec.max_radius_bins)
        self.n_radius_bins_draw = int(spec.draw_radius_bins)
        self.n_radius_bins_nondraw = int(spec.nondraw_radius_bins)
        self.uses_flag_specific_radius_bins = self.n_radius_bins_draw != self.n_radius_bins_nondraw
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.angle_offset = 3
        self.radius_offset = self.angle_offset + self.n_angle_bins
        if self.spec.scheme == "polar_2token":
            self.radflag_offset = self.radius_offset
            self.draw_offset = None
            if self.uses_flag_specific_radius_bins:
                self.vocab_size = self.radflag_offset + self.n_radius_bins_nondraw + self.n_radius_bins_draw
            else:
                self.vocab_size = self.radflag_offset + self.n_radius_bins_draw * 2
        elif self.spec.scheme == "polar_3token":
            self.radflag_offset = None
            self.draw_offset = self.radius_offset + self.n_radius_bins_draw
            self.vocab_size = self.draw_offset + 2
        else:
            raise ValueError(f"Unsupported tokenizer scheme: {self.spec.scheme}")

    def radius_bin_count(self, draw_flag: int) -> int:
        return self.n_radius_bins_draw if int(draw_flag) == 1 else self.n_radius_bins_nondraw

    def angle_token(self, angle_bin: int) -> int:
        return self.angle_offset + int(angle_bin)

    def radflag_token(self, radius_bin: int, draw_flag: int) -> int:
        if self.radflag_offset is None:
            raise ValueError("radflag_token is only valid for polar_2token")
        radius_bin = int(radius_bin)
        draw_flag = int(draw_flag)
        max_bins = self.radius_bin_count(draw_flag)
        if radius_bin < 0 or radius_bin >= max_bins:
            raise ValueError(f"radius_bin {radius_bin} is out of range for draw_flag={draw_flag}")
        if not self.uses_flag_specific_radius_bins:
            return self.radflag_offset + radius_bin * 2 + draw_flag
        if draw_flag == 0:
            return self.radflag_offset + radius_bin
        return self.radflag_offset + self.n_radius_bins_nondraw + radius_bin

    def radius_token(self, radius_bin: int) -> int:
        return self.radius_offset + int(radius_bin)

    def draw_token(self, draw_flag: int) -> int:
        if self.draw_offset is None:
            raise ValueError("draw_token is only valid for polar_3token")
        return self.draw_offset + int(draw_flag)

    def is_angle_token(self, token_id: int) -> bool:
        return self.angle_offset <= token_id < self.radius_offset

    def is_radflag_token(self, token_id: int) -> bool:
        if self.radflag_offset is None:
            return False
        return self.radflag_offset <= token_id < self.vocab_size

    def is_radius_token(self, token_id: int) -> bool:
        if self.draw_offset is None:
            return False
        return self.radius_offset <= token_id < self.draw_offset

    def is_draw_token(self, token_id: int) -> bool:
        if self.draw_offset is None:
            return False
        return self.draw_offset <= token_id < self.vocab_size

    def decode_angle_bin(self, token_id: int) -> int:
        return token_id - self.angle_offset

    def decode_radflag(self, token_id: int) -> Tuple[int, int]:
        if self.radflag_offset is None:
            raise ValueError("decode_radflag is only valid for polar_2token")
        value = token_id - self.radflag_offset
        if not self.uses_flag_specific_radius_bins:
            radius_bin = value // 2
            draw_flag = value % 2
            return radius_bin, draw_flag
        if value < self.n_radius_bins_nondraw:
            return value, 0
        value -= self.n_radius_bins_nondraw
        if value < self.n_radius_bins_draw:
            return value, 1
        raise ValueError(f"Invalid radflag token id: {token_id}")

    def decode_radius_bin(self, token_id: int) -> int:
        return token_id - self.radius_offset

    def decode_draw_flag(self, token_id: int) -> int:
        if self.draw_offset is None:
            raise ValueError("decode_draw_flag is only valid for polar_3token")
        return token_id - self.draw_offset


class PolarOffsetTokenizer:
    def __init__(
        self,
        spec: TokenizerSpec,
        radius_data: np.ndarray | RadiusCodebooks,
        stroke_vocab: StrokeVocab,
    ):
        self.spec = spec
        self.n_angle_bins = int(spec.n_angle_bins)
        self.stroke_vocab = stroke_vocab
        if isinstance(radius_data, RadiusCodebooks):
            self.radius_codebooks = radius_data
        else:
            radius_edges = np.asarray(radius_data, dtype=np.float32)
            if self.stroke_vocab.uses_flag_specific_radius_bins:
                raise ValueError("Legacy shared radius_edges input does not support asymmetric draw/non-draw bins")
            if radius_edges.ndim != 1 or len(radius_edges) < 2:
                raise ValueError("radius_edges must be a 1D array with at least 2 values")
            if len(radius_edges) != self.stroke_vocab.radius_bin_count(1) + 1:
                raise ValueError("radius_edges length must be n_radius_bins + 1")
            midpoints = 0.5 * (radius_edges[:-1] + radius_edges[1:])
            shared = RadiusCodebook(
                edges=radius_edges,
                midpoints=np.asarray(midpoints, dtype=np.float32),
                medians=np.asarray(midpoints, dtype=np.float32),
                decode_values=np.asarray(midpoints, dtype=np.float32),
                sample_count=0,
            )
            self.radius_codebooks = RadiusCodebooks(
                mode="shared",
                decode_mode="midpoint",
                shared=shared,
            )
        self.radius_edges = self.radius_codebooks.legacy_radius_edges()
        expected_bins = [
            (self.radius_codebooks.shared, self.stroke_vocab.radius_bin_count(1)),
            (self.radius_codebooks.nondraw, self.stroke_vocab.radius_bin_count(0)),
            (self.radius_codebooks.draw, self.stroke_vocab.radius_bin_count(1)),
        ]
        for codebook, n_bins in expected_bins:
            if codebook is None:
                continue
            if codebook.edges.ndim != 1 or len(codebook.edges) < 2:
                raise ValueError("radius codebook edges must be a 1D array with at least 2 values")
            if len(codebook.edges) != n_bins + 1:
                raise ValueError("radius codebook edges length does not match tokenizer radius bin count")

    def encode_offsets(self, offsets: np.ndarray) -> List[int]:
        if offsets.ndim != 2 or offsets.shape[1] < 3:
            raise ValueError(f"Expected offsets shape [N,3+] but got {offsets.shape}")

        dx = offsets[:, 0].astype(np.float32)
        dy = offsets[:, 1].astype(np.float32)
        draw_flag = np.clip(np.rint(offsets[:, 2].astype(np.float32)), 0, 1).astype(np.int64)

        theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        angle_bins = np.floor(theta / (2 * np.pi) * self.n_angle_bins).astype(np.int64)
        angle_bins = np.clip(angle_bins, 0, self.n_angle_bins - 1)

        radius = np.sqrt(dx * dx + dy * dy)
        radius_bins = np.empty_like(draw_flag)
        if self.radius_codebooks.mode == "shared":
            radius_bins = np.searchsorted(self.radius_edges, radius, side="right") - 1
            radius_bins = np.clip(radius_bins, 0, self.stroke_vocab.radius_bin_count(1) - 1)
        else:
            for flag in (0, 1):
                mask = draw_flag == flag
                if not np.any(mask):
                    continue
                edges = self.radius_codebooks.codebook_for_flag(flag).edges
                bins = np.searchsorted(edges, radius[mask], side="right") - 1
                radius_bins[mask] = np.clip(bins, 0, self.stroke_vocab.radius_bin_count(flag) - 1)

        tokens: List[int] = []
        for angle_bin, radius_bin, flag in zip(angle_bins, radius_bins, draw_flag):
            tokens.append(self.stroke_vocab.angle_token(int(angle_bin)))
            if self.spec.scheme == "polar_2token":
                tokens.append(self.stroke_vocab.radflag_token(int(radius_bin), int(flag)))
            elif self.spec.scheme == "polar_3token":
                tokens.append(self.stroke_vocab.radius_token(int(radius_bin)))
                tokens.append(self.stroke_vocab.draw_token(int(flag)))
            else:
                raise ValueError(f"Unsupported tokenizer scheme: {self.spec.scheme}")
        tokens.append(self.stroke_vocab.eos_id)
        return tokens

    def decode_tokens_to_offsets(self, tokens: Sequence[int]) -> np.ndarray:
        filtered = [t for t in tokens if t not in (self.stroke_vocab.pad_id, self.stroke_vocab.bos_id)]
        rows: List[List[float]] = []
        if not filtered:
            return np.zeros((0, 3), dtype=np.float32)

        if self.spec.scheme == "polar_2token":
            current_angle: Optional[int] = None
            for token_id in filtered:
                if token_id == self.stroke_vocab.eos_id:
                    break
                if self.stroke_vocab.is_angle_token(token_id):
                    current_angle = self.stroke_vocab.decode_angle_bin(token_id)
                elif self.stroke_vocab.is_radflag_token(token_id) and current_angle is not None:
                    radius_bin, draw_flag = self.stroke_vocab.decode_radflag(token_id)
                    theta = ((current_angle + 0.5) / self.n_angle_bins) * (2 * np.pi)
                    radius = float(self.radius_codebooks.codebook_for_flag(draw_flag).decode_values[radius_bin])
                    dx = radius * math.cos(theta)
                    dy = radius * math.sin(theta)
                    rows.append([dx, dy, float(draw_flag)])
                    current_angle = None
        elif self.spec.scheme == "polar_3token":
            current_angle: Optional[int] = None
            current_radius_bin: Optional[int] = None
            for token_id in filtered:
                if token_id == self.stroke_vocab.eos_id:
                    break
                if self.stroke_vocab.is_angle_token(token_id):
                    current_angle = self.stroke_vocab.decode_angle_bin(token_id)
                    current_radius_bin = None
                elif self.stroke_vocab.is_radius_token(token_id) and current_angle is not None:
                    current_radius_bin = self.stroke_vocab.decode_radius_bin(token_id)
                elif (
                    self.stroke_vocab.is_draw_token(token_id)
                    and current_angle is not None
                    and current_radius_bin is not None
                ):
                    draw_flag = self.stroke_vocab.decode_draw_flag(token_id)
                    theta = ((current_angle + 0.5) / self.n_angle_bins) * (2 * np.pi)
                    radius = float(self.radius_codebooks.codebook_for_flag(draw_flag).decode_values[current_radius_bin])
                    dx = radius * math.cos(theta)
                    dy = radius * math.sin(theta)
                    rows.append([dx, dy, float(draw_flag)])
                    current_angle = None
                    current_radius_bin = None
        else:
            raise ValueError(f"Unsupported tokenizer scheme: {self.spec.scheme}")
        return np.asarray(rows, dtype=np.float32)


def resolve_tokenizer_spec(tokenizer_variant: Optional[str], config_path: Optional[Path]) -> TokenizerSpec:
    if config_path is None:
        variant_name = tokenizer_variant or "baseline_2token_64x32"
        if variant_name not in BUILTIN_TOKENIZER_SPECS:
            raise KeyError(f"Unknown tokenizer variant '{variant_name}'")
        return _validate_tokenizer_spec(BUILTIN_TOKENIZER_SPECS[variant_name])

    raw = json.loads(config_path.read_text())
    scheme = str(raw.get("scheme") or "polar_2token")
    name = str(raw.get("name") or raw.get("variant_name") or tokenizer_variant or config_path.stem)
    if "n_angle_bins" not in raw or "n_radius_bins" not in raw:
        raise KeyError(f"Tokenizer config {config_path} must define n_angle_bins and n_radius_bins")
    return _validate_tokenizer_spec(TokenizerSpec(
        name=name,
        scheme=scheme,
        n_angle_bins=int(raw["n_angle_bins"]),
        n_radius_bins=int(raw["n_radius_bins"]),
        max_points_for_bins=int(raw.get("max_points_for_bins", 500_000)),
        radius_codebook_mode=str(raw.get("radius_codebook_mode") or "shared"),
        radius_decode_mode=str(raw.get("radius_decode_mode") or "midpoint"),
        n_radius_bins_draw=(
            int(raw["n_radius_bins_draw"])
            if raw.get("n_radius_bins_draw") is not None
            else None
        ),
        n_radius_bins_nondraw=(
            int(raw["n_radius_bins_nondraw"])
            if raw.get("n_radius_bins_nondraw") is not None
            else None
        ),
    ))


def collect_radius_edges_from_offsets(
    offsets_array: Sequence[np.ndarray],
    n_radius_bins: int,
    max_points_for_bins: int,
    seed: int,
) -> np.ndarray:
    spec = TokenizerSpec(
        name="legacy_radius_edges",
        scheme="polar_2token",
        n_angle_bins=64,
        n_radius_bins=n_radius_bins,
        max_points_for_bins=max_points_for_bins,
    )
    codebooks = collect_radius_codebooks_from_offsets(
        offsets_array,
        spec=spec,
        max_points_for_bins=max_points_for_bins,
        seed=seed,
    )
    return codebooks.legacy_radius_edges()
