# app/services/generation.py
import torch
import numpy as np
from pathlib import Path

from utils import plot_stroke
from utils.data_utils import valid_offset_normalization, data_denormalization
from generate import generate_conditional_sequence
from app.core.singletons import ModelSingleton, VocabSingleton, StatsSingleton
from app.services.job_store import create_job, mark_sample_done, complete_job, fail_job


def run_generation_job(
    job_id: str,
    job_dir: Path,
    char_seq: str,
    style_path: Path,
    priming_text: str,
    bias: float,
    n_samples: int,
):
    """Runs in a background thread. Writes PNGs as each sample finishes."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    create_job(job_id, job_dir, n_samples)

    try:
        model = ModelSingleton._model
        char_to_id = VocabSingleton.char_to_id
        train_mean = StatsSingleton.train_mean
        train_std = StatsSingleton.train_std

        style = np.load(
            str(style_path), allow_pickle=True, encoding="bytes"
        ).astype(np.float32)

        # FIX: normalize with GLOBAL training stats, not local style stats
        style_norm = valid_offset_normalization(train_mean, train_std, style.copy())
        style_tensor = torch.from_numpy(style_norm).unsqueeze(0).to(device)

        for i in range(n_samples):
            gen_seq, _ = generate_conditional_sequence(
                model,
                char_seq,
                device,
                char_to_id,
                VocabSingleton.idx_to_char,
                bias,
                prime=True,
                prime_seq=style_tensor,
                real_text=priming_text,
                is_map=False,
            )
            gen_seq = data_denormalization(train_mean, train_std, gen_seq)
            sample_path = job_dir / f"sample_{i}.png"
            plot_stroke(gen_seq[0], str(sample_path))
            mark_sample_done(job_id, job_dir, i + 1)

        complete_job(job_id, job_dir)

    except Exception as e:
        fail_job(job_id, job_dir, str(e))
        raise
