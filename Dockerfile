FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.fastembed
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-download the embedding and reranker models so the container starts fast.
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"
COPY src ./src
COPY static ./static
COPY data/index ./data/index
COPY data/manifest.json ./data/manifest.json
ENV PYTHONPATH=/app/src
EXPOSE 7860
# Set GROQ_API_KEY (and optionally GROQ_MODEL) as a secret/environment variable on your host.
CMD ["uvicorn", "ragdemo.api:app", "--host", "0.0.0.0", "--port", "7860"]
