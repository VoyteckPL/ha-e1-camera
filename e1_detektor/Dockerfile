FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install onnxruntime numpy pillow aiohttp opencv-python-headless

WORKDIR /app

RUN python -c "import urllib.request as u; b='https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'; [u.urlretrieve(b+n+'.onnx', '/app/'+n+'.onnx') for n in ('yolox_s','yolox_tiny')]"

# model odciskow aut: MobileNetV2 (onnx model zoo) z dodatkowym wyjsciem = wektor tuz przed klasyfikatorem; brak modelu = tylko kolor
RUN pip install onnx \
    && (python -c "import urllib.request as u, onnx; p='/tmp/m.onnx'; u.urlretrieve('https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx', p); m=onnx.load(p); n=[x for x in m.graph.node if x.op_type=='Gemm'][-1]; m.graph.output.append(onnx.helper.make_tensor_value_info(n.input[0], onnx.TensorProto.FLOAT, None)); onnx.save(m, '/app/emb.onnx')" || echo 'BRAK MODELU ODCISKOW') \
    ; pip uninstall -y onnx; rm -f /tmp/m.onnx

COPY app.py strefy.py reolink.py auta.py bufor.py tuner.py /app/
COPY kamera_panel.html /app/kamera_panel.html

CMD ["python", "/app/app.py"]
