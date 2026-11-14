FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN pip install numpy pillow paho-mqtt requests tzdata

WORKDIR /app
COPY app.py kompozycja.py /app/

CMD ["python", "/app/app.py"]
