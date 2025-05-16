FROM python:3.12-slim

# 1. Set working directory
WORKDIR /app

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy source code
COPY . .

# 4. Launch Gunicorn, pointing at the flask_app in main.py
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "main:flask_app"]

