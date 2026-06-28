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
# Stage 1: compile dsacstar C++ extension against system OpenCV + PyTorch
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 20.04 EOL (abr 2025): archive.ubuntu.com y security.ubuntu.com ya no sirven.
# Apuntar a old-releases.ubuntu.com y descartar los componentes -updates/-security/-backports.
RUN sed -ri 's|https?://([a-z]{2}\.)?archive.ubuntu.com|https://old-releases.ubuntu.com|g; \
             s|https?://security.ubuntu.com|https://old-releases.ubuntu.com|g' \
        /etc/apt/sources.list \
    && sed -i '/-updates\|-security\|-backports/d' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3.9-dev python3-pip \
        build-essential libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --no-cache-dir \
        torch==2.0.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

COPY dsacstar/ /build/dsacstar/
WORKDIR /build/dsacstar

RUN sed -i "s|opencv_inc_dir = ''|opencv_inc_dir = '/usr/include/opencv4'|" setup.py && \
    sed -i "s|opencv_lib_dir = ''|opencv_lib_dir = '/usr/lib/x86_64-linux-gnu'|" setup.py && \
    python3.9 setup.py build_ext --inplace

# ---------------------------------------------------------------------------
# Stage 2: runtime image with ACE + compiled dsacstar (no toolchain)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 20.04 EOL (mismo workaround que builder)
RUN sed -ri 's|https?://([a-z]{2}\.)?archive.ubuntu.com|https://old-releases.ubuntu.com|g; \
             s|https?://security.ubuntu.com|https://old-releases.ubuntu.com|g' \
        /etc/apt/sources.list \
    && sed -i '/-updates\|-security\|-backports/d' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.9 python3-pip \
        libopencv-core4.2 libopencv-calib3d4.2 \
        libgomp1 \
        libglu1-mesa libglvnd0 libgl1 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Todo el stack Python en una capa: torch (cu118) + resto desde PyPI
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
        PyOpenGL \
        numpy==1.23.5 scipy==1.10.0 scikit-image==0.19.3 \
        opencv-python-headless matplotlib trimesh pyrender pillow==9.4.0 pyyaml

# Usuario no-root (UID/GID configurables vía build-arg; default 1000:1000)
ARG UID=1000
ARG GID=1000
RUN groupadd -f -g ${GID} appuser && useradd -u ${UID} -g ${GID} -m -s /bin/bash appuser

# Extensión C++ RANSAC compilada (linkea contra libopencv-* de arriba)
COPY --from=builder --chown=${UID}:${GID} /build/dsacstar/dsacstar*.so /ace/

# Código ACE + pesos del encoder
COPY --chown=${UID}:${GID} *.py /ace/
COPY --chown=${UID}:${GID} ace_encoder_pretrained.pt /ace/
COPY --chown=${UID}:${GID} datasets/ /ace/datasets/
COPY --chown=${UID}:${GID} scripts/ /ace/scripts/

WORKDIR /ace
USER ${UID}:${GID}

ENTRYPOINT ["python"]
