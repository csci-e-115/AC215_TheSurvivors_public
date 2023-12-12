
:: create environment variables
set IMAGE_NAME=03-train-model
set BASE_DIR=%cd%
set SECRETS_DIR=%cd%\..\secrets
set GCS_BUCKET_NAME="ac215-dermaid"
set GCP_PROJECT="ac215-dermaid"
set GCP_REGION="us-central1"
set GCS_MODELS="models"
set GCS_DATASETS="datasets"
set GCS_AUGMENTED_TRAINING_DATA="train_data"
set GCS_VALIDATION_DATA="validation_data"
set /p GIT_LOGIN=<..\secrets\git.txt
set DVC_TAG="dataset_v09"
set NUM_GPU=1
set /p WANDB_API_KEY=<..\secrets\wandb.txt

:: Build image from dockerfile
docker build -t %IMAGE_NAME% -f Dockerfile .

:: Run container
docker run --rm --gpus all --name %IMAGE_NAME% -ti ^
-v "%SECRETS_DIR%":/secrets ^
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json ^
 %IMAGE_NAME% %GCP_PROJECT% %GCS_BUCKET_NAME% %GIT_LOGIN% %WANDB_API_KEY% %DVC_TAG% %NUM_GPU%
