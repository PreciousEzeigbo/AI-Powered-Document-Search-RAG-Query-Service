FROM python:3.12-slim

# Install system dependencies required by PyMuPDF, sqlite3, etc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    sqlite3 \
    cargo \
    rustc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for blazingly fast dependency installation
RUN pip install --no-cache-dir uv

# Copy uv dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into system environment so Docker doesn't need venv activation
RUN uv pip install --system -r pyproject.toml

# Copy project files
COPY . .

# Set environment variables for production networking
ENV ENV=prod
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
