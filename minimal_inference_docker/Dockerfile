FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get --no-install-recommends install -y  \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
  imagecodecs \
  numpy \
  pillow \
  scikit-image \
  scipy \
  tensorflow-cpu==2.13.0 \
  tifffile

RUN python3 -c "import tensorflow as tf"

RUN mkdir /app && mkdir /data

WORKDIR /app

COPY main.py /app/main.py
COPY fcnn_c.keras /app/fcnn_c.keras

CMD ["python3", "main.py"]