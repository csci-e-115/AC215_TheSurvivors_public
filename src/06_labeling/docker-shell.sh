#!/bin/bash

set -e

# Create the network if we don't have it yet
docker network inspect dermaid-data-labeling-network >/dev/null 2>&1 || docker network create dermaid-data-labeling-network

# Build the image based on the Dockerfile
# docker build -t 06-dermaid-data-label-cli --platform=linux/arm64/v8 -f Dockerfile .
docker build -t 06-dermaid-data-label-cli -f Dockerfile .

# Run All Containers
docker-compose run --rm --service-ports 06-dermaid-data-label-cli