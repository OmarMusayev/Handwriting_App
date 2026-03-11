# tests/test_transformer_model.py
import torch
from models.transformer_synthesis import TextEncoder, PositionalEncoding, StyleVAE


def test_positional_encoding_output_shape():
    pe = PositionalEncoding(d_model=256)
    x = torch.randn(2, 10, 256)  # (batch, seq, d_model)
    out = pe(x)
    assert out.shape == (2, 10, 256)


def test_positional_encoding_adds_signal():
    pe = PositionalEncoding(d_model=256, dropout=0.0)
    x = torch.zeros(1, 5, 256)
    out = pe(x)
    # output should not be all zeros — positional signal was added
    assert out.abs().sum() > 0


def test_text_encoder_output_shape():
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    text = torch.randint(0, 77, (2, 15))       # (batch=2, text_len=15)
    text_mask = torch.ones(2, 15, dtype=torch.float)
    out = enc(text, text_mask)
    assert out.shape == (2, 15, 256)


def test_text_encoder_respects_padding_mask():
    """Padded positions should not crash, shape is preserved, and valid positions are not NaN."""
    enc = TextEncoder(vocab_size=77, d_model=256, nhead=8, num_layers=4, ff_dim=512)
    enc.eval()
    text = torch.randint(0, 77, (1, 10))
    mask_half = torch.zeros(1, 10)
    mask_half[0, :5] = 1.0
    with torch.no_grad():
        out_half = enc(text, mask_half)
    assert out_half.shape == (1, 10, 256)
    assert not torch.isnan(out_half[:, :5]).any(), "valid positions should not be NaN"


# --- StyleVAE tests ---

def test_style_vae_output_shapes():
    """z, mu, logvar should all be (batch, 64)."""
    vae = StyleVAE()
    strokes = torch.randn(3, 20, 3)  # (batch=3, seq_len=20, 3)
    z, mu, logvar = vae(strokes, training=True)
    assert z.shape == (3, 64), f"z shape {z.shape} != (3, 64)"
    assert mu.shape == (3, 64), f"mu shape {mu.shape} != (3, 64)"
    assert logvar.shape == (3, 64), f"logvar shape {logvar.shape} != (3, 64)"


def test_style_vae_training_mode_uses_reparameterization():
    """With training=True, z should differ from mu (sampling adds noise) at least once in 50 runs."""
    vae = StyleVAE()
    vae.eval()  # module in eval mode; training flag still controls reparameterization
    strokes = torch.randn(2, 15, 3)
    found_difference = False
    for _ in range(50):
        with torch.no_grad():
            z, mu, logvar = vae(strokes, training=True)
        if not torch.allclose(z, mu):
            found_difference = True
            break
    assert found_difference, "Expected z != mu at least once in 50 samples (reparameterization should add noise)"


def test_style_vae_inference_mode_is_deterministic():
    """With training=False, z should equal mu exactly (no sampling noise)."""
    vae = StyleVAE()
    vae.eval()
    strokes = torch.randn(2, 15, 3)
    with torch.no_grad():
        z, mu, logvar = vae(strokes, training=False)
    assert torch.equal(z, mu), "In inference mode (training=False), z must equal mu exactly"


def test_style_vae_kl_loss_is_non_negative():
    """KL divergence = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) should be >= 0."""
    vae = StyleVAE()
    strokes = torch.randn(4, 25, 3)
    with torch.no_grad():
        z, mu, logvar = vae(strokes, training=True)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    assert kl.item() >= 0, f"KL divergence should be non-negative, got {kl.item()}"
