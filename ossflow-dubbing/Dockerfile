# syntax=docker/dockerfile:1.7
# Build context: ../python

# ----- Stage 1: s2.cpp CUDA builder -------------------------------------
# Uses NVIDIA's CUDA 12.4 devel image on Ubuntu 22.04 (GLIBC 2.35 — matches
# bjj-base CUDA 12.4 runtime so the binary loads cleanly in the runtime
# stage).
#
# Why CUDA and not Vulkan: Docker Desktop on Windows/WSL2 does NOT inject
# the NVIDIA Vulkan ICD into containers (only CUDA passes through), so a
# Vulkan-built s2 crashes with `vk::IncompatibleDriverError` in WSL2-hosted
# dev. CUDA gives us dev/prod parity (workstation WSL2 + Proxmox LXC
# server both use CUDA), keeps all-in on NVIDIA tooling, and avoids the
# `graphics` capability dance that nvidia-container-toolkit doesn't fully
# honor on Docker Desktop.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS s2cpp-builder
ARG S2CPP_REPO=https://github.com/rodrigomatta/s2.cpp
ARG S2CPP_REF=e48ce8e02d8335bd9a0ba94679f605724b31d123
# Compute capability for Turing (RTX 2070S workstation, RTX 2060 server).
# Both are sm_75. Adding sm_86 would also cover Ampere if you later move
# to a 3060/3090, but it doubles compile time.
ARG CUDA_ARCHITECTURES=75
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules ${S2CPP_REPO} /src/s2.cpp \
 && cd /src/s2.cpp && git checkout ${S2CPP_REF} \
 && git submodule update --init --recursive

WORKDIR /src/s2.cpp
# libcuda.so.1 (CUDA driver API) lives in the NVIDIA driver injected at
# runtime by nvidia-container-toolkit, not in the cuda-devel image. The
# devel image ships a link-time stub at /usr/local/cuda/lib64/stubs/
# named `libcuda.so` (no .1 suffix). When `s2` is linked, ld follows the
# transitive NEEDED `libcuda.so.1` from libggml-cuda.so and fails because
# that exact filename doesn't exist in the stubs dir. Symlink the .1
# alias so the linker resolves the SONAME during build; the real
# libcuda.so.1 from the driver overrides at runtime.
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DS2_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build --parallel "$(nproc)" \
 && echo "=== Build artifacts ===" \
 && find /src/s2.cpp/build -maxdepth 4 -name "s2" -type f -executable -exec ls -la {} + \
 && find /src/s2.cpp/build -name "libggml*.so*" -exec ls -la {} +

# ----- Stage 2: dubbing-generator runtime -------------------------------
FROM bjj-base:latest

# CUDA runtime libs come from bjj-base (which is built on
# nvidia/cuda:12.4-runtime). No Vulkan, no extra graphics tooling needed.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        rubberband-cli wget espeak-ng \
 && rm -rf /var/lib/apt/lists/*

# Verified actual artifact paths from successful build:
#   /src/s2.cpp/build/s2                              -- the executable
#   /src/s2.cpp/build/ggml/src/libggml{,-base,-cpu}.so.* + symlinks
#   /src/s2.cpp/build/ggml/src/ggml-cuda/libggml-cuda.so.* + symlinks
COPY --from=s2cpp-builder /src/s2.cpp/build/s2 /usr/local/bin/s2
COPY --from=s2cpp-builder /src/s2.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
COPY --from=s2cpp-builder /src/s2.cpp/build/ggml/src/ggml-cuda/libggml-cuda*.so* /usr/local/lib/
RUN ldconfig \
 && echo "=== Verifying s2 binary linkage ===" \
 && ls -la /usr/local/bin/s2 /usr/local/lib/libggml*.so* \
 && ldd /usr/local/bin/s2 | grep -E "ggml|cuda" || true
# Note: `libcuda.so.1 => not found` in `ldd` output during build is
# EXPECTED — that lib comes from the NVIDIA driver injected by
# nvidia-container-toolkit at runtime. We cannot run `s2 --help` in
# build to smoke-test because `bjj-base` doesn't ship the driver stub
# in the runtime path. Runtime verification happens via the FastAPI
# startup hook (which logs s2.cpp stdout).

COPY bjj_service_kit /app/bjj_service_kit
COPY dubbing-generator/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt \
 && pip install -e /app/bjj_service_kit \
 && pip uninstall -y pytest pytest-asyncio || true \
 && find /opt/venv -depth -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete \
 && find /opt/venv/lib/python3.11/site-packages -maxdepth 3 -type d \
      \( -name "tests" -o -name "test" -o -name "docs" -o -name "examples" \) \
      -exec rm -rf {} + 2>/dev/null || true

RUN mkdir -p /models/piper && cd /models/piper && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx && \
    wget -q https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json

# Pre-cache Kokoro-82M (kept as fallback engine, not the default).
RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='e')" || true

COPY dubbing-generator /app/dubbing-generator
WORKDIR /app/dubbing-generator

# HF cache still used by torch hub (silero-vad en VoiceCloner) y demucs.
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8003/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
