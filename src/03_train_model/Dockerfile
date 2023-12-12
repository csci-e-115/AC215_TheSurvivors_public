# Use the official Debian-hosted Python image
# FROM python:3.8-slim-buster
FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest

ARG DEBIAN_PACKAGES="build-essential git curl"
# Prevent apt from showing prompts
ENV DEBIAN_FRONTEND=noninteractive

# Tell pipenv where the shell is.
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash
# ENV GOOGLE_APPLICATION_CREDENTIALS=secrets/ac215-dermaid.json

# Ensure we have an up to date baseline, install dependencies
RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app_base

WORKDIR /app_base

# Add Pipfile, Pipfile.lock
ADD Pipfile Pipfile.lock ./

RUN pipenv install

ADD run-commands.sh run-training.sh ./

RUN chmod u+x run-training.sh

RUN git config --global --add safe.directory /app_base

# Entry point
ENTRYPOINT ["/bin/bash", "run-commands.sh"]

