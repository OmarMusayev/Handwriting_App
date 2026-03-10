# Contributing to Hand Magic

Thanks for your interest! Here's how to get started.

## Setup

```bash
git clone https://github.com/your-handle/hand-magic.git
cd hand-magic
pip install -e ".[dev]"
cp .env.example .env
uvicorn main:app --reload
```

## Running tests

```bash
pytest -v
```

## Guidelines

- Keep PRs focused — one feature or bug fix per PR
- Add tests for new behaviour
- Run `pytest` before submitting
- Open an issue first for large changes

## Model weights

`data/strokes.npy`, `data/sentences.txt`, and `results/synthesis/*.pt` are included in the repo.
They come from the IAM On-Line Handwriting Database pre-processed by the original author.
