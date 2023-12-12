set IMAGE_NAME="11-run-kfp"
set BASE_DIR=%cd%
set SECRETS_DIR=%cd%\..\secrets
set GCP_PROJECT="ac215-dermaid"
set GCS_BUCKET_NAME="ac215-dermaid"
set GCP_REGION="us-east1"
set GCS_SERVICE_ACCOUNT="data-service-account-dermaid@ac215-dermaid.iam.gserviceaccount.com"
set DVC_TAG="dataset_v09"
set /p WANDB_API_KEY=< %SECRETS_DIR%\wandb.txt
set NUM_GPU="4"
set WHYLABS_DEFAULT_ORG_ID="org-dPEWaJ"
set /p WHYLABS_API_KEY=< %SECRETS_DIR%\whylabs.txt
set WHYLABS_DEFAULT_DATASET_ID="dermaid"

:: Build the image based on the Dockerfile
docker build -t %IMAGE_NAME% -f Dockerfile .
::docker build -t %IMAGE_NAME% --platform=linux/amd64 -f Dockerfile .

:: Run Container
docker run --rm --name %IMAGE_NAME% -ti ^
-v "%BASE_DIR%":/app ^
-v "%SECRETS_DIR%":/secrets ^
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json ^
-e GCP_PROJECT=%GCP_PROJECT% ^
-e GCP_REGION=%GCP_REGION% ^
-e GCS_BUCKET_NAME=%GCS_BUCKET_NAME% ^
-e GCS_SERVICE_ACCOUNT=%GCS_SERVICE_ACCOUNT% ^
-e WANDB_API_KEY=%WANDB_API_KEY% ^
-e WHYLABS_DEFAULT_ORG_ID=%WHYLABS_DEFAULT_ORG_ID% ^
-e WHYLABS_API_KEY=%WHYLABS_API_KEY% ^
-e WHYLABS_DEFAULT_DATASET_ID=%WHYLABS_DEFAULT_DATASET_ID% ^
-e DVC_TAG=%DVC_TAG% ^
-e NUM_GPU=%NUM_GPU% ^
--env-file git.env ^
%IMAGE_NAME%