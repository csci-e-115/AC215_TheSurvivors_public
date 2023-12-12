
:: create environment variables
set BASE_DIR=%cd%
set SECRETS_DIR=%cd%\..\secrets

set GCP_PROJECT="ac215-dermaid"
set GCS_BUCKET_NAME="ac215-dermaid"
set GCP_ZONE="us-central1-a"
set WHYLABS_DEFAULT_ORG_ID="org-dPEWaJ"
set /p WHYLABS_API_KEY=< %SECRETS_DIR%\whylabs.txt
set WHYLABS_DEFAULT_DATASET_ID="dermaid"

:: Build image from dockerfile
docker build -t 02-preprocess-data -f Dockerfile .

:: Run container
docker run --rm --name 02-preprocess-data -ti ^
-v "%BASE_DIR%\..\..":/app_base ^
-v "%SECRETS_DIR%":/secrets ^
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json ^
-e GCP_PROJECT=%GCP_PROJECT% ^
-e GCP_ZONE=%GCP_ZONE% ^
-e GCS_BUCKET_NAME=%GCS_BUCKET_NAME% ^
-e WHYLABS_DEFAULT_ORG_ID=%WHYLABS_DEFAULT_ORG_ID% ^
-e WHYLABS_API_KEY=%WHYLABS_API_KEY% ^
-e WHYLABS_DEFAULT_DATASET_ID=%WHYLABS_DEFAULT_DATASET_ID% ^
02-preprocess-data