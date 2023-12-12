#!/bin/bash

set -e

# create environment variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_ZONE="us-central1-a"

# Build image from dockerfile
docker build -t 01-get-raw-data -f Dockerfile .

# Run container
docker run --rm --name 01-get-raw-data -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
01-get-raw-data