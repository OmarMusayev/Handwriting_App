"""
Task 11: Full integration smoke test for Transformer + Style VAE pipeline.

Tests the end-to-end pipeline without requiring real data files.
"""
import inspect
import numpy as np
import pytest
import torch
from unittest.mock import patch, MagicMock

from models.transformer_synthesis import HandWritingSynthesisTransformer
from generate import generate_conditional_sequence
from app.core.singletons import ModelSingleton
from app.core.config import settings


# ---------------------------------------------------------------------------
# Test 1: Full pipeline forward pass
# ---------------------------------------------------------------------------

def test_full_pipeline_forward_pass():
    """Instantiate the model and run a full forward pass, verify output shapes."""
    model = HandWritingSynthesisTransformer(vocab_size=10)
    model.eval()

    strokes = torch.randn(2, 8, 3)
    text = torch.randint(0, 10, (2, 5))
    text_mask = torch.ones(2, 5)
    style = torch.randn(2, 4, 3)

    with torch.no_grad():
        y_hat, mu, logvar = model(strokes, text, text_mask, style, use_sampling=False)

    assert y_hat.shape == (2, 8, 121), f"Expected (2, 8, 121), got {y_hat.shape}"
    assert mu.shape == (2, 64), f"Expected (2, 64), got {mu.shape}"
    assert logvar.shape == (2, 64), f"Expected (2, 64), got {logvar.shape}"


# ---------------------------------------------------------------------------
# Test 2: Full pipeline generate
# ---------------------------------------------------------------------------

def test_full_pipeline_generate():
    """Run model.generate() and verify returned numpy array shape."""
    model = HandWritingSynthesisTransformer(vocab_size=10)
    model.eval()

    text = torch.tensor([[1, 2, 3]])
    text_mask = torch.ones(1, 3)
    style = torch.randn(1, 5, 3)

    result = model.generate(text, text_mask, style, bias=1.0, max_steps=10)

    assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"
    assert result.shape[0] == 1, f"Expected batch dim=1, got {result.shape[0]}"
    assert result.shape[2] == 3, f"Expected stroke dim=3, got {result.shape[2]}"
    assert result.shape[1] > 0, "Expected at least one generated step"
    assert result.shape[1] <= 10, f"Expected T <= max_steps=10, got {result.shape[1]}"


# ---------------------------------------------------------------------------
# Test 3: generate_conditional_sequence integration with Transformer
# ---------------------------------------------------------------------------

class FakeTransformer(HandWritingSynthesisTransformer):
    """Subclass so isinstance(..., HandWritingSynthesisTransformer) is True."""

    def __init__(self):
        # Skip the real __init__ to avoid needing config params
        pass

    def generate(self, text, text_mask, style_strokes, bias, max_steps):
        return np.zeros((1, 5, 3))

    def eval(self):
        return self


def test_generate_conditional_sequence_transformer_integration():
    """Call generate_conditional_sequence() with a Transformer model instance."""
    model = FakeTransformer()

    gen_seq, phi = generate_conditional_sequence(
        model_or_path=model,
        char_seq="hi",
        device="cpu",
        char_to_id={"h": 0, "i": 1, " ": 2},
        idx_to_char=lambda x: [str(i) for i in x],
        bias=1.0,
        prime=True,
        prime_seq=np.zeros((5, 3), dtype=np.float32),
        real_text="hi",
        is_map=False,
    )

    assert phi == [], f"Expected empty phi for transformer, got {phi}"
    assert isinstance(gen_seq, np.ndarray), f"Expected numpy array, got {type(gen_seq)}"
    assert gen_seq.shape == (1, 5, 3), f"Expected (1, 5, 3), got {gen_seq.shape}"
    assert gen_seq.shape[2] == 3, f"Expected stroke dim=3, got {gen_seq.shape[2]}"


def test_generate_conditional_sequence_transformer_none_prime_seq():
    """When prime_seq=None, the adapter should create a zero style tensor (1, 1, 3)."""
    import generate as gen_mod

    captured_style = {}

    class CapturingFakeTransformer(HandWritingSynthesisTransformer):
        def __init__(self):
            pass

        def generate(self, text, text_mask, style_strokes, bias, max_steps):
            captured_style["style_strokes"] = style_strokes
            return np.zeros((1, 3, 3))

        def eval(self):
            return self

    model = CapturingFakeTransformer()
    gen_seq, phi = generate_conditional_sequence(
        model_or_path=model,
        char_seq="hi",
        device="cpu",
        char_to_id={"h": 0, "i": 1, " ": 2},
        idx_to_char=lambda x: [str(i) for i in x],
        bias=1.0,
        prime=False,
        prime_seq=None,
        real_text="",
        is_map=False,
    )

    style = captured_style["style_strokes"]
    assert style.shape == (1, 1, 3), f"Expected zeros(1,1,3), got {style.shape}"
    assert style.sum().item() == 0.0, "Expected all-zero style strokes for None prime_seq"
    assert phi == []


# ---------------------------------------------------------------------------
# Test 4 (revised): ModelSingleton returns transformer model in eval mode
# ---------------------------------------------------------------------------

def test_model_singleton_serves_transformer():
    """ModelSingleton.get() with model_type='transformer' returns eval-mode model."""
    import app.core.singletons as singletons_mod
    import models.transformer_synthesis as tsyn_mod

    mock_model = MagicMock()
    mock_model.load_state_dict = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)

    with patch.object(tsyn_mod, "HandWritingSynthesisTransformer", return_value=mock_model):
        with patch("torch.load", return_value={"model_state": {}}):
            singletons_mod.ModelSingleton._model = None
            result = singletons_mod.ModelSingleton.get(
                "fake.pt", "cpu", 10, model_type="transformer"
            )

    assert result is mock_model, "ModelSingleton.get() should return the mock model"
    mock_model.eval.assert_called_once()


# ---------------------------------------------------------------------------
# Test 5: Config model_type is used by startup_singletons
# ---------------------------------------------------------------------------

def test_config_model_type_used_in_startup():
    """settings.model_type is valid and startup_singletons accepts model_type param."""
    assert settings.model_type in ("lstm", "transformer"), (
        f"Unexpected model_type value: {settings.model_type}"
    )

    from app.core.singletons import startup_singletons
    sig = inspect.signature(startup_singletons)
    assert "model_type" in sig.parameters, (
        "startup_singletons() must accept a model_type parameter"
    )
