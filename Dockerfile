FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock

COPY data ./data
COPY docs ./docs
COPY src ./src
COPY static ./static
COPY entrypoint.sh ./

RUN mkdir -p artifacts && chmod +x entrypoint.sh \
    && useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PYTHONPATH=src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"]

CMD ["./entrypoint.sh"]
