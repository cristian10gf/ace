# syntax=docker/dockerfile:1
#
# ACE (Accelerated Coordinate Encoding) Docker image.
# Builds the dsacstar C++ extension and packages ACE for training/evaluation.
#
# Build context: preprocesamiento/models/ace/
#   docker build -f docker/Dockerfile -t ace:latest .
#
# The encoder weights (ace_encoder_pretrained.pt) should be in the build
# context root. Download from:
#   https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz

# ---------------------------------------------------------------------------
# Stage 1: compile dsacstar C++ extension
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_PREFIX=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        build-essential \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --no-cache-dir \
        torch==2.0.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

COPY dsacstar/ /build/dsacstar/
WORKDIR /build/dsacstar

RUN sed -i "s|opencv_inc_dir = ''|opencv_inc_dir = '/usr/include/opencv4'|" setup.py && \
    sed -i "s|opencv_lib_dir = ''|opencv_lib_dir = '/usr/lib/x86_64-linux-gnu'|" setup.py && \
    python3.8 setup.py build_ext --inplace

# ---------------------------------------------------------------------------
# Stage 2: runtime image with ACE + compiled dsacstar
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3-pip \
        libopencv-core4.2 \
        libopencv-calib3d4.2 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN python -m pip install --no-cache-dir \
        torch==2.0.0+cu118 \
        torchvision==0.15.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

RUN python -m pip install --no-cache-dir \
        numpy==1.23.5 \
        scipy==1.10.0 \
        scikit-image==0.19.3 \
        opencv-python-headless \
        matplotlib \
        trimesh \
        pyrender \
        pillow==9.4.0 \
        pyyaml

# Compiled C++ RANSAC extension
COPY --from=builder /build/dsacstar/dsacstar*.so /ace/

# ACE source code
COPY *.py /ace/
COPY datasets/ /ace/datasets/
COPY scripts/ /ace/scripts/

WORKDIR /ace

ENTRYPOINT ["python"]
