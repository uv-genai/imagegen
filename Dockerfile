# syntax=docker/dockerfile:1
#
# imagegen — OpenAI-compatible image generation gateway.
#
# Multi-stage: the pinned stable-diffusion.cpp engine is built twice, once per
# compute backend, and both binaries ship in the final image. Which one runs is a
# runtime choice (IMAGEGEN_BACKEND), not a rebuild.
#
#   docker build -t imagegen:latest .
#   docker run --rm --gpus all -e IMAGEGEN_API_KEYS=... -v imagegen-models:/models imagegen:latest
#
# Weights are never baked into the image: they are ~31.7 GB and change
# independently of the code. Mount a volume at /models and use the `fetch-models`
# entrypoint command to populate it.

ARG CUDA_VERSION=13.3.1

# ---------------------------------------------------------------------------
# Engine, CUDA backend.
# The build script owns the pinned commit and the submodule verification, so the
# image builds exactly the engine the repository pins — no drift.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS engine-cuda

# The build container has no GPU, so nvcc cannot use '-arch=native'. Pin the
# target: 75 = Turing (RTX 20-series). Widen for distribution, e.g.
# "75;80;86;89;90" — each extra architecture multiplies the CUDA compile time.
ARG CUDA_ARCHITECTURES=75
ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY scripts/build_engine.sh scripts/build_engine.sh
RUN scripts/build_engine.sh cuda \
    && mkdir -p /artifacts \
    && cp -a third_party/stable-diffusion.cpp/build/bin /artifacts/bin

# ---------------------------------------------------------------------------
# Engine, Vulkan backend.
# Two packages here are easy to miss, and the build fails without them:
#   glslc         — the engine does find_package(Vulkan COMPONENTS glslc REQUIRED),
#                   and Ubuntu's glslang-tools does NOT provide glslc.
#   spirv-headers — the engine does find_package(SPIRV-Headers CONFIG REQUIRED);
#                   this package ships /usr/share/cmake/SPIRV-Headers.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS engine-vulkan

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates glslc libvulkan-dev spirv-headers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY scripts/build_engine.sh scripts/build_engine.sh
RUN scripts/build_engine.sh vulkan \
    && mkdir -p /artifacts \
    && cp -a third_party/stable-diffusion.cpp/build-vulkan/bin /artifacts/bin

# ---------------------------------------------------------------------------
# Runtime.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime

# libvulkan1  : Vulkan loader.
# libgomp1    : GNU OpenMP runtime, a NEEDED entry of the engine binaries but not
#               present in the CUDA runtime base image.
# libegl1     : required. The NVIDIA Vulkan ICD dlopens libEGL.so.1 at runtime;
#               it is not a NEEDED entry, so `ldd` will not show it missing and
#               the failure surfaces only as
#               "Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'".
# libx11-6/libxext6 : the ICD's other runtime dependencies.
# libglx0     : GLVND GLX dispatch pulled in by the same path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates \
        libvulkan1 libgomp1 libegl1 libglx0 libx11-6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 comes from Ubuntu 24.04 itself, so the interpreter and every system
# library share one glibc — no cross-distro copying of a Python install.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1

# uv supplies `uvx`, which scripts/fetch_models.sh uses to run the Hugging Face
# CLI without adding it to the service's runtime dependency graph.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && pip install --no-cache-dir uv

COPY --from=engine-cuda   /artifacts/bin /opt/imagegen/engine/cuda/bin
COPY --from=engine-vulkan /artifacts/bin /opt/imagegen/engine/vulkan/bin
COPY scripts/fetch_models.sh /opt/imagegen/scripts/fetch_models.sh
COPY config/container.toml /etc/imagegen/config.toml
COPY docker/entrypoint.sh /usr/local/bin/imagegen-entrypoint

# Default engine; the entrypoint repoints this on IMAGEGEN_BACKEND.
RUN ln -sfn /opt/imagegen/engine/cuda /opt/imagegen/engine/current \
    && chmod +x /usr/local/bin/imagegen-entrypoint /opt/imagegen/scripts/fetch_models.sh

WORKDIR /models
EXPOSE 8080
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["/usr/local/bin/imagegen-entrypoint"]
CMD ["serve"]
