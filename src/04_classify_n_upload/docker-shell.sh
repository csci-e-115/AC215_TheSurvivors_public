#!/bin/bash

set -e 

# create environment variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_ZONE="us-central1-a"
export GCS_MODELS="models"
export RAW_IMAGES_PATH="unlabelled_raw_images"
export METADATA_PATH="unlabelled_metadata"
export PREPROC_IMAGES_PATH="unlabelled_preproc_images"

# build image
docker build -t artexmg/mockapp_prediction:latest .

docker run --rm --name mockapp_prediction -ti \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME  \
-e GCS_MODELS=$GCS_MODELS \
-e RAW_IMAGES_PATH=$RAW_IMAGES_PATH \
-e METADATA_PATH=$METADATA_PATH \
-e PREPROC_IMAGES_PATH=$PREPROC_IMAGES_PATH \
-e PORT="8080" \
-p 8080:8080 \
artexmg/mockapp_prediction:latest