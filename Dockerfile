# RunPod Serverless worker: headless Blender weight painting.
# Blender (bone-heat) is CPU + RAM bound — no GPU/CUDA needed. Size the RunPod
# endpoint for RAM (>=32 GB) rather than VRAM.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Shared libraries Blender needs even in --background mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl xz-utils ca-certificates \
    libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxext6 \
    libgl1 libegl1 libsm6 libice6 libglib2.0-0 libxkbcommon0 \
    libxrandr2 libxinerama1 libxcursor1 libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Bake Blender into the image so cold starts don't download/install anything.
# (Pair this with FlashBoot on the endpoint for the fastest warm starts.)
ARG BLENDER_SERIES=4.2
ARG BLENDER_VER=4.2.3
RUN curl -fsSL "https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VER}-linux-x64.tar.xz" -o /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && /opt/blender/blender --version

ENV BLENDER_BIN=/opt/blender/blender
ENV AUTOWEIGHT_SCRIPT=/app/blender_autoweight.py

# The principled harmonic weight solver needs scipy/numpy INSIDE Blender's
# bundled Python (not the system python). Install with Blender's own python so
# wheels match its CPython ABI exactly, then verify Blender can import them at
# build time (fail the build if not). If this layer is ever removed, the
# autoweight script simply falls back to bone-heat.
RUN PYBIN="$(ls /opt/blender/${BLENDER_SERIES}/python/bin/python3* | head -n1)" \
    && echo "Blender python: $PYBIN" \
    && ("$PYBIN" -m ensurepip --upgrade || true) \
    && "$PYBIN" -m pip install --no-cache-dir --upgrade pip \
    && "$PYBIN" -m pip install --no-cache-dir scipy \
    && /opt/blender/blender --background --python-expr "import numpy, scipy; print('blender-scipy-ok', numpy.__version__, scipy.__version__)"

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY blender_autoweight.py /app/blender_autoweight.py
COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
