FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends -y \
    build-essential curl wget git sox ffmpeg libsndfile1 zip unzip mandoc groff

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --root-user-action=ignore --no-cache-dir --no-deps -r requirements.txt

ARG VERSION
ENV VERSION=$VERSION
