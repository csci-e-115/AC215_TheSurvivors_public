#!/bin/bash

set -e

# create environment variables
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/

export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_ZONE="us-central1-a"
export GCS_DATASETS="datasets"
export GCS_MODELS="models"
export GCS_AUGMENTED_TRAINING_DATA="train_data"
export GCS_VALIDATION_DATA="validation_data"
export DVC_TAG="dataset_v09"
export NUM_GPU=1
export GIT_LOGIN=$(head -1 "../secrets/git.txt")
WANDB_API_KEY=$(head -1 "../secrets/wandb.txt")

# Build image from dockerfile
docker build -t 03-train-model -f Dockerfile .

# Run container
docker run --rm --name 03-train-model -ti \
-v "$SECRETS_DIR":/secrets/ \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
03-train-model $GCP_PROJECT $GCS_BUCKET_NAME $GIT_LOGIN $WANDB_API_KEY $DVC_TAG $NUM_GPU
