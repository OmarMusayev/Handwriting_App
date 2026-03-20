# app/services/transformer_generation.py
import random
from pathlib import Path

from app.core.transformer_singleton import TransformerSingleton
from app.services.job_store import create_job, mark_sample_done, complete_job, fail_job


def run_transformer_generation_job(
    job_id: str,
    job_dir: Path,
    text: str,
    n_samples: int,
):
    """Background thread: generates n_samples via IAMOnDB transformer (top-k=20)."""
    create_job(job_id, job_dir, n_samples)

    try:
        from handwriting.generation import save_sample_plot

        for i in range(n_samples):
            seed = random.randint(0, 2**31 - 1)
            points = TransformerSingleton.generate_sample(text, seed=seed)

            sample_path = job_dir / f"sample_{i}.png"
            save_sample_plot(sample_path, points)
            mark_sample_done(job_id, job_dir, i + 1)

        complete_job(job_id, job_dir)

    except Exception as e:
        fail_job(job_id, job_dir, str(e))
        raise
