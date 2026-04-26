FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /workspace/ru-moonshine

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace/ru-moonshine
