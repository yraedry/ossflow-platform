# syntax=docker/dockerfile:1.7

# ----- Stage 1: s2.cpp Vulkan builder -----------------------------------
# Uses NVIDIA's CUDA 12.4 devel image on Ubuntu 22.04 as base — we don't
# need CUDA toolkit for the build, but matching ossflow-base's GLIBC 2.35
# and runtime libs keeps the binary loading cleanly downstream.
#
# Why Vulkan and not CUDA: ggml's CUDA backend does NOT support
# `get_rows` for K-quant tensors (Q4_K, Q5_K, Q6_K). With s2.cpp pinned
# at e48ce8e the CUDA path silently falls back to CPU compute on a
# Q4_K_M model, which collapses synthesis to ~77 s/phrase on a 2060.
# Vulkan has full K-quant kernel coverage and runs the same Q4_K_M model
# at GPU speed. Target deployment is a Linux LXC on Proxmox where the
# NVIDIA driver injects the Vulkan ICD via nvidia-container-toolkit's
# `graphics` capability — no WSL2/Docker Desktop in the loop.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS s2cpp-builder
ARG S2CPP_REPO=https://github.com/rodrigomatta/s2.cpp
ARG S2CPP_REF=e48ce8e02d8335bd9a0ba94679f605724b31d123
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libvulkan-dev glslang-tools glslang-dev spirv-tools \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --recurse-submodules ${S2CPP_REPO} /src/s2.cpp \
 && cd /src/s2.cpp && git checkout ${S2CPP_REF} \
 && git submodule update --init --recursive

WORKDIR /src/s2.cpp
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DS2_VULKAN=ON \
 && cmake --build build --parallel "$(nproc)" \
 && echo "=== Build artifacts ===" \
 && find /src/s2.cpp/build -maxdepth 4 -name "s2" -type f -executable -exec ls -la {} + \
 && find /src/s2.cpp/build -name "libggml*.so*" -exec ls -la {} +

# ----- Stage 2: dubbing-generator runtime -------------------------------
FROM ossflow-base:latest

# Vulkan runtime: libvulkan1 is the loader; mesa-vulkan-drivers ships
# fallback ICDs (lavapipe) used when no GPU is visible (CI/tests). The
# real NVIDIA Vulkan ICD is injected at container start by
# nvidia-container-toolkit when the service is launched with
# NVIDIA_DRIVER_CAPABILITIES including `graphics` (see compose).
# vulkan-tools provides `vulkaninfo` for runtime debugging.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        rubberband-cli wget espeak-ng \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
 && rm -rf /var/lib/apt/lists/*

# Verified actual artifact paths from successful build:
#   /src/s2.cpp/build/s2                                -- the executable
#   /src/s2.cpp/build/ggml/src/libggml{,-base,-cpu}.so.* + symlinks
#   /src/s2.cpp/build/ggml/src/ggml-vulkan/libggml-vulkan.so.* + symlinks
COPY --from=s2cpp-builder /src/s2.cpp/build/s2 /usr/local/bin/s2
COPY --from=s2cpp-builder /src/s2.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
COPY --from=s2cpp-builder /src/s2.cpp/build/ggml/src/ggml-vulkan/libggml-vulkan*.so* /usr/local/lib/
RUN ldconfig \
 && echo "=== Verifying s2 binary linkage ===" \
 && ls -la /usr/local/bin/s2 /usr/local/lib/libggml*.so* \
 && ldd /usr/local/bin/s2 | grep -E "ggml|vulkan" || true
# Note: the NVIDIA Vulkan ICD (libGLX_nvidia, libnvidia-glcore, the
# /usr/share/vulkan/icd.d/nvidia_icd.json manifest) is injected at
# container start by nvidia-container-toolkit when the service requests
# the `graphics` capability — it is NOT baked into this image. Inside a
# running container, `vulkaninfo --summary` should list the RTX 2060.
# If it lists `llvmpipe` instead, the driver injection failed (check
# NVIDIA_DRIVER_CAPABILITIES in compose).

COPY ossflow-dubbing/requirements.txt /app/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt \
 && pip install --no-cache-dir "ossflow-service-kit @ git+https://github.com/yraedry/ossflow-core@v0.1.0#subdirectory=ossflow_service_kit" \
 && pip uninstall -y pytest pytest-asyncio || true \
 && find /opt/venv -depth -type d -name "__pycache__" -exec rm -rf {} + \
 && find /opt/venv -type f -name "*.pyc" -delete \
 && find /opt/venv/lib/python3.11/site-packages -maxdepth 3 -type d \
      \( -name "tests" -o -name "test" -o -name "docs" -o -name "examples" \) \
      -exec rm -rf {} + 2>/dev/null || true

# Pre-cache de modelos eliminado tras T22.5: Piper y Kokoro ya no están
# soportados como motores TTS. S2-Pro usa los .gguf montados como volumen
# (ver docker-compose.yml: /models/s2pro montado desde el host) — no se
# bakean en la imagen para que el usuario pueda cambiar entre q4_k_m y
# q6_k sin rebuild.

COPY ossflow-dubbing /app/ossflow-dubbing
WORKDIR /app/ossflow-dubbing

# HF cache still used by torch hub (silero-vad en VoiceCloner) y demucs.
ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8003/health || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
