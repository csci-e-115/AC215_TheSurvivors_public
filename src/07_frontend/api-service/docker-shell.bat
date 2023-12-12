REM Define some environment variables
SET IMAGE_NAME="dermaid-app-api-server"
# Define some environment variables
set IMAGE_NAME="dermaid-app-api-service"
set BASE_DIR=%cd%
set SECRETS_DIR=%BASE_DIR%\..\..\secrets\
set PERSISTENT_DIR=%BASE_DIR%\..\..\persistent-folder\
set GCS_BUCKET_NAME="dermaid-app-models"

REM Build the image based on the Dockerfile
:: docker build -t %IMAGE_NAME% -f Dockerfile .

REM Run the container
:: cd ..
:: -e GCS_BUCKET_NAME=%GCS_BUCKET_NAME%
:: --mount type=bind,source=%PERSISTEN_DIR%,target=/persistent
docker run  --rm --name %IMAGE_NAME% -ti ^
            --mount type=bind,source=%BASE_DIR%,target=/app ^
            --mount type=bind,source=%SECRETS_DIR%,target=/secrets ^
            --mount type=bind,source=%PERSISTENT_DIR%,target=/persistent ^
            -p 9000:9000 ^
            -e DEV=1 ^
            -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/ac215-dermaid.json ^
            %IMAGE_NAME%