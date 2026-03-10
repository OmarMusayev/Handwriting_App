FROM python:3.12-slim

WORKDIR /app

# Install system deps for matplotlib/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
