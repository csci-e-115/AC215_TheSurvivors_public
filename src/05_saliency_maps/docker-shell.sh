#!/bin/bash
set -e 

# create environment variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_ZONE="us-central1-a"
export GCS_MODELS="models"
export GCS_DATASETS="datasets"
export GCS_AUGMENTED_TRAINING_DATA="augmented_training_data"
export GCS_SALIENCY_MAPS="saliency_maps"
export GCS_IMAGES_UPLOAD="uploads_images"
export GCS_JSON_UPLOAD="uploads"

# build image
docker build -t artexmg/mockapp_saliency_maps:latest .

docker run --rm --name mockapp_saliency_maps -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME  \
-e GCS_MODELS=$GCS_MODELS \
-e GCS_AUGMENTED_TRAINING_DATA=$GCS_AUGMENTED_TRAINING_DATA \
-e GCS_SALIENCY_MAPS=$GCS_SALIENCY_MAPS \
-e GCS_IMAGES_UPLOAD=$GCS_IMAGES_UPLOAD \
-e GCS_JSON_UPLOAD=$GCS_JSON_UPLOAD \
-p 8181:8181 \
-e PORT="8181" \
artexmg/mockapp_saliency_maps:latest