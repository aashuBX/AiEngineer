FROM python:3.12-slim

WORKDIR /app

# System dependencies (psycopg2 build deps, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    default-libmysqlclient-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Install uv and dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system . pymysql

# Shared data volume for file system tool
RUN mkdir -p /app/data/shared

EXPOSE 8002

CMD ["uvicorn", "src.backend_api.app:app", "--host", "0.0.0.0", "--port", "8002"]
