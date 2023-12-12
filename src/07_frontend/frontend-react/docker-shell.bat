SET IMAGE_NAME=07-frontend
SET BASE_DIR=%cd%

:: docker build -t %IMAGE_NAME% -f Dockerfile.dev .
:: docker run --rm --name %IMAGE_NAME% -ti --mount type=bind,source="%cd%",target=/app -p 3000:3000 %IMAGE_NAME%

docker build -t %IMAGE_NAME% -f Dockerfile .
:: docker run --rm --name %IMAGE_NAME% -ti -p 3000:3000 %IMAGE_NAME%
docker run --rm --name %IMAGE_NAME% -ti -p 80:80 %IMAGE_NAME%
