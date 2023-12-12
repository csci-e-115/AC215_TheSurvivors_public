
:: create environment variables
set BASE_DIR=%cd%
set SECRETS_DIR=%cd%\..\secrets

set GCP_PROJECT="ac215-dermaid"
set GCS_BUCKET_NAME="ac215-dermaid"
set GCP_ZONE="us-central1-a"
set GCS_MODELS="models"
set RAW_IMAGES_PATH="unlabelled_raw_images"
set METADATA_PATH="unlabelled_metadata"
set PREPROC_IMAGES_PATH="unlabelled_preproc_images"

:: Build image from dockerfile
:: docker build -t 04_classify_n_upload -f Dockerfile .

:: Run container
echo $BASE_DIR
echo $SECRETS_DIR
docker run --rm --name mockapp_prediction -ti ^
-v "%BASE_DIR%":/app ^
-v "%SECRETS_DIR%":/secrets ^
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json ^
-e GCP_PROJECT=%GCP_PROJECT% ^
-e GCP_ZONE=%GCP_ZONE% ^
-e GCS_BUCKET_NAME=%GCS_BUCKET_NAME% ^
-e GCS_MODELS=%GCS_MODELS% ^
-e RAW_IMAGES_PATH=%RAW_IMAGES_PATH% ^
-e METADATA_PATH=%METADATA_PATH% ^
-e PREPROC_IMAGES_PATH=%PREPROC_IMAGES_PATH% ^
-e PORT="8080" ^
-p 8080:8080 ^
04_classify_n_upload