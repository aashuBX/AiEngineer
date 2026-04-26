FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system . fastapi uvicorn[standard]

# Create data directory for SQLite checkpoints
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
