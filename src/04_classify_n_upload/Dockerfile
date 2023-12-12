# Use the official Tensorflow image.
FROM tensorflow/tensorflow


# Set the working directory in Docker
WORKDIR /app

# Copy the local code to the container
COPY ./mock_app_prediction.py  ./requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run on boot. Here we use the "gunicorn" production server.
CMD gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT mock_app_prediction:app
