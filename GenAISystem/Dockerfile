FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install uv and dependencies
# Note: sentence-transformers downloads model weights on first run
# They are cached in /root/.cache/huggingface
RUN pip install --no-cache-dir uv && \
    uv pip install --system . pinecone-client

# Create data directory
RUN mkdir -p /app/data

EXPOSE 8001

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
