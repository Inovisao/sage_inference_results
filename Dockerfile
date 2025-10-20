FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL maintainer="Codex CLI" \
      description="Inference pipeline for Sage project (YOLOv8, Faster R-CNN, YOLOv5-TPH)."

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY detectors/YOLOV5_TPH/tph-yolov5/requirements.txt /tmp/requirements-yolov5-tph.txt
RUN pip install -r /tmp/requirements-yolov5-tph.txt && rm /tmp/requirements-yolov5-tph.txt

COPY . .

ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "run_pipeline.py"]
