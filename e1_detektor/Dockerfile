FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install onnxruntime numpy pillow aiohttp

WORKDIR /app

RUN python -c "import urllib.request as u; b='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'; [u.urlretrieve(b+n+'.onnx', '/app/'+n+'.onnx') for n in ('yolox_s','yolox_tiny')]"

COPY app.py strefy.py reolink.py auta.py /app/
COPY kamera_panel.html /app/kamera_panel.html

CMD ["python", "/app/app.py"]
