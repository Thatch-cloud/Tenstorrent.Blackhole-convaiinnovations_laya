FROM python:3.11-slim-bookworm@sha256:4b4c524dc3dce996864e030c7bd9c6b0e517597189fee48f48e05b499442444b

ARG SOURCE_REVISION
LABEL org.opencontainers.image.title="Laya CPU reference service" \
      org.opencontainers.image.source="https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya" \
      org.opencontainers.image.revision=$SOURCE_REVISION
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface TORCHINDUCTOR_CACHE_DIR=/tmp/laya-torch-cache \
    XDG_CACHE_HOME=/tmp/laya-cache GIT_OPTIONAL_LOCKS=0
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates passwd \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 laya \
    && useradd --uid 10001 --gid 10001 --home-dir /var/lib/laya \
       --no-create-home --shell /usr/sbin/nologin laya
WORKDIR /opt/laya
COPY pyproject.toml ./
COPY src ./src
COPY configs/checkpoint-lock.json ./configs/checkpoint-lock.json
COPY tests/fixtures/cpu-reference ./tests/fixtures/cpu-reference
COPY tests/test_cpu_runtime.py ./tests/test_cpu_runtime.py
COPY scripts/download_checkpoint.py ./scripts/download_checkpoint.py
RUN python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir numpy==2.4.6 safetensors==0.8.0 transformers==5.17.0 \
       tokenizers==0.23.2 huggingface-hub==1.16.1 '.[serve]' \
    && python -m pip check
RUN git init .cache/upstream/laya \
    && git -C .cache/upstream/laya remote add origin https://github.com/NandhaKishorM/laya.git \
    && git -C .cache/upstream/laya fetch --depth 1 origin 970dc8c5f63d7b886a68409493f37d569424f933 \
    && git -C .cache/upstream/laya checkout --detach FETCH_HEAD \
    && git config --system --add safe.directory /opt/laya/.cache/upstream/laya \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/download_checkpoint.py \
    && python -m pip freeze > /opt/laya/environment.lock \
    && dpkg-query -W > /opt/laya/os-packages.lock \
    && mkdir -p /var/lib/laya && chown 10001:10001 /var/lib/laya
USER 10001:10001
ENTRYPOINT ["laya-cpu-serve", "--root", "/opt/laya"]
