# Hosted build of the lawbureaucracy brain (search-only).
#
# The image stays small on purpose: the 450 MB index and the 1.1 GB bge-m3 ONNX
# model are NOT baked in — bootstrap.py pulls them onto the mounted volume on
# first boot, so redeploys are fast and don't re-ship a gigabyte.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    EMBED_BACKEND=onnx \
    ANSWER_ENGINE=openai_compat \
    BOOTSTRAP=1 \
    HOST=0.0.0.0

WORKDIR /app

COPY brain/requirements-cloud.txt brain/requirements-cloud.txt
RUN pip install -r brain/requirements-cloud.txt

COPY brain/ brain/
COPY web/ web/

# Python puts the script's own dir on sys.path, so brain/'s flat imports resolve.
CMD ["python", "brain/server.py"]
