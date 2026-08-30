# ---- builder: install deps into a venv we can copy wholesale ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch: the GPU wheels are ~2.5 GB and this service runs on CPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" "torchvision>=0.17"

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

# ---- runtime ----
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/home/app/.cache/torch

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app kb/ ./kb/
COPY --chown=app:app artifacts/ ./artifacts/

USER app
EXPOSE 8000

# Bake the ImageNet backbone into the image so a cold container does not pull
# 330 MB on its first request — that download is the difference between a 2 s
# and a 90 s first response.
RUN python -c "from torchvision import models; models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)"

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=4).status==200 else 1)"

CMD ["uvicorn", "freshsense.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
