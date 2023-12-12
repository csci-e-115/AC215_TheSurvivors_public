from google.cloud import storage
from google.oauth2 import service_account
import os, io

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

# Quantization
import tensorflow_model_optimization as tfmot

# Enable/Disable Eager Execution
# Reference: https://www.tensorflow.org/guide/eager
# TensorFlow's eager execution is an imperative programming environment
# that evaluates operations immediately, without building graphs

tf.compat.v1.disable_eager_execution()
tf.compat.v1.enable_eager_execution()

print(f"tensorflow version {tf.__version__}")
print(f"keras version {tf.keras.__version__}")
print(f"Eager Execution Enabled: {tf.executing_eagerly()}\n")

devices = tf.config.get_visible_devices()
print(f"All Devices: \n{devices}\n")
print(f"Available GPUs: \n{tf.config.list_logical_devices('GPU')}\n")

# Better performance with the tf.data API
# Reference: https://www.tensorflow.org/guide/data_performance
AUTOTUNE = tf.data.AUTOTUNE

# Set GCS variables
bucket_name='ac215-dermaid'
bucket_name='ac215-dermaid-test'
models_folder="models"
augmented_training_data_folder="datasets/augmented_training_data"


dataset_dir = f"gs://{bucket_name}/{augmented_training_data_folder}/"
model_dir = f"gs://{bucket_name}/{models_folder}/"

# Load regular (big) model
trained_model = tf.keras.models.load_model(model_dir)

# trained_model = tf.keras.models.load_model(model_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)

tflite_model = converter.convert()

print(f"Quantized model length {len(tflite_model)}")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Assest reduction between big and quantized model
len(tflite_quant_model)/len(tflite_model)


# saves quantized model to be used 
with open('tflite_quant_model.tflite','wb') as f: 
    f.write(tflite_quant_model)

# METHOD 2: Quantization Aware Training

# Doesn't work due to custome layer
# TA Recommendation: do not develop a custom code, but find somthing already build

# quantize_model = tfmot.quantization.keras.quantize_model
# q_aware_model = quantize_model(trained_model)

# opt = keras.optimizers.Adam(learning_rate=0.0001)
# q_aware_model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ['categorical_accuracy'])
