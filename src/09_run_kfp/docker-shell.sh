export IMAGE_NAME="11-run-kfp"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../secrets/
export GCP_PROJECT="ac215-dermaid"
export GCS_BUCKET_NAME="ac215-dermaid"
export GCP_REGION="us-east1"
export GCS_SERVICE_ACCOUNT="data-service-account-dermaid@ac215-dermaid.iam.gserviceaccount.com"
export DVC_TAG="dataset_v09"
export WANDB_API_KEY=$(head -1 "../secrets/wandb.txt")
export NUM_GPU="4"
export WHYLABS_DEFAULT_ORG_ID="org-fEdKE7"
export WHYLABS_API_KEY="7gZspSDGKK.LqWlJLqFz9Y2gTlH0ry1EQGOqqbSMNk69vvZBfo4VsmztgRTMbJbn:org-fEdKE7"
export WHYLABS_DEFAULT_DATASET_ID="model-1"

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .
# docker build -t $IMAGE_NAME --platform=linux/amd64 -f Dockerfile .

# Run Container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_REGION=$GCP_REGION \
-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
-e GCS_SERVICE_ACCOUNT=$GCS_SERVICE_ACCOUNT \
-e WANDB_API_KEY=$WANDB_API_KEY \
-e WHYLABS_DEFAULT_ORG_ID=$WHYLABS_DEFAULT_ORG_ID \
-e WHYLABS_API_KEY=$WHYLABS_API_KEY \
-e WHYLABS_DEFAULT_DATASET_ID=$WHYLABS_DEFAULT_DATASET_ID \
-e NUM_GPU=$NUM_GPU \
--env-file git.env \
$IMAGE_NAME

# -e DVC_TAG=$DVC_TAG \
