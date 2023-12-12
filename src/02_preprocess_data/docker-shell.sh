#!/bin/bash

set -e

# create environment variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_ZONE="us-central1-a"
export WHYLABS_DEFAULT_ORG_ID="org-dPEWaJ"
export WHYLABS_API_KEY=$(head -1 "../secrets/whylabs.txt")
export WHYLABS_DEFAULT_DATASET_ID="dermaid"

# Build image from dockerfile
docker build -t 02-preprocess-data -f Dockerfile .

# Run container
docker run --rm --name 02-preprocess-data -ti \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e WHYLABS_DEFAULT_ORG_ID=$WHYLABS_DEFAULT_ORG_ID \
-e WHYLABS_API_KEY=$WHYLABS_API_KEY \
-e WHYLABS_DEFAULT_DATASET_ID=$WHYLABS_DEFAULT_DATASET_ID \
02-preprocess-data
