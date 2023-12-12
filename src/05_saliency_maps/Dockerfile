# Use the official tensorflow image.
FROM tensorflow/tensorflow

ARG DEBIAN_PACKAGES="build-essential git curl"
# Tell pipenv where the shell is.
# This allows us to use "pipenv shell" as a container entry point.
ENV PYENV_SHELL=/bin/bash

RUN set -ex; \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
    apt-get clean && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    mkdir -p /app/

# Set the working directory in Docker
WORKDIR /app

# Add Pipfile, Pipfile.lock + python code
ADD requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code to the container
ADD . /app/

# ENTRYPOINT ["/bin/bash", "run-server.sh"]

# # Set the command to run on boot. Here we use the "gunicorn" production server.
CMD gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 600 mock_app_saliency_maps:app
