ARG REGISTRY=docker.io/library
FROM $REGISTRY/nvidia/cuda:11.3.1-base-ubuntu20.04

ADD docker/requirements-pip.txt requirements-pip.txt

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --no-install-recommends \
        gcc \
        python-is-python3 \
        python3 \
        python3-pip \
        python3-dev \
        libgtk2.0-dev \
        libxtst6 \
        tree \
        vim \
        wget \
        zip \
        && \
        rm -rf /var/lib/apt/lists/*

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Europe/Moscow apt-get -y install tzdata
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade --requirement requirements-pip.txt
RUN python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN python3 -m pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
RUN python3 -m pip install mmdet==2.16.0

COPY . /deltadist
WORKDIR /deltadist

ENV PYTHONPATH="${PYTHONPATH}:/deltadist/"
