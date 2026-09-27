# Render hosts only the coordinator. Compute runs on your workers.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api/cloud.py api/storage.py api/routing.py api/datasets.py ./api/
COPY worker/ ./worker/
COPY benchmarks/datasets ./benchmarks/datasets
COPY benchmarks/reports ./benchmarks/reports
COPY examples/models ./examples/models
RUN useradd --uid 10001 --create-home sovereign && mkdir -p /var/data && chown sovereign:sovereign /var/data
USER sovereign
ENV SOVEREIGN_DB=/var/data/workspace.db
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn api.cloud:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips='*'"]
