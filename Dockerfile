FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="Codex CLI" \
      description="SAGE SAHI inference pipeline for YOLOV8 and YOLOV11."

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "run_inference.py"]
