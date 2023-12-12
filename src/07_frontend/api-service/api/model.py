import os
import json
import requests
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_hub as hub
from google.cloud import aiplatform
import base64
import time

class_names = {
    'MEL' : "Melanoma",
    'NV'  : "MelanocyticNevus",
    'BCC' : "BasalCellCarcinoma",
    'AK'  : "ActinicKeratosis",
    'BKL' : "BeningKeratosis",
    'DF'  : "Dermatofibroma",
    'VASC': "VascularLesion",
    'SCC' : "SquamousCellCarcinoma",
}

malignant_lesion = [ "Melanoma",
                 "BasalCellCarcinoma",
                 "ActinicKeratosis",
                 "SquamousCellCarcinoma"]

AUTOTUNE = tf.data.experimental.AUTOTUNE
local_experiments_path = "/persistent/experiments"
best_model = None
best_model_id = None
prediction_model = None
data_details = None
image_width = 224
image_height = 224
num_channels = 3


def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    image_width = 224
    image_height = 224
    num_channels = 3

    # Prepare the data
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
    test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data


def make_prediction(image_path, metadata):
    # Load & preprocess
    test_data = load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = prediction_model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = data_details["index2label"][str(idx)]

    if prediction_model.layers[-1].activation.__name__ != "softmax":
        prediction = tf.nn.softmax(prediction).numpy()
        print(prediction)

    malignant = False
    if prediction_label in malignant_lesion:
       malignant = True


    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction) * 100, 2),
        "malignant": malignant,
    }


def make_request_with_retries(url, data, files, max_retries=3, delay=1):
        '''
            Make the API request with retries
        '''
        for attempt in range(max_retries):
            response = requests.post(url, data=data, files=files)
            
            if response.status_code == 200:
                return response  # Success, return the response
            else:
                print(f"Attempt {attempt + 1} failed with status code {response.status_code}. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait for a bit before retrying

        return None  # If all attempts fail, return None

def make_prediction_cloud_run(image_path, metadata):
    '''
    Call to CloudRun API for prediction 
    '''
    # API Endpoint hosted on Cloud Run (DermAID)
    saliency_url="https://dermaid-saliency-maps-qhg6oc4tja-uc.a.run.app/gradcam_post/"


    image_name = os.path.basename(image_path)
 
    files={
        'image':(image_name, open(image_path, 'rb'), 'image/jpeg')
    } 
    
    # Manage retrival mode
    retries_active = False
    
    # Making a POST request to the API
    if retries_active:
        response = make_request_with_retries(saliency_url, files=files, data=metadata)
    else:
        response = requests.post(saliency_url, files=files, data=metadata)

    # Verifying response
    if response.status_code == 200:
        print("Request successfull.")

        response_data = response.json()

        prediction_label = response_data.get('prediction_name')
        prediction = response_data.get('prediction_proba')
        accuracy = round(np.max(prediction) * 100, 2)
        saliency_map_url = response_data.get('saliency_map_url')
        # Set type of lesion according to prediction label
        malignant = False
        if prediction_label in malignant_lesion:
            malignant = True
 

        print(f'Image [{image_name}], Diagnostic: [{prediction_label}] with probability: [{prediction}], malignant: {malignant}')

        return {
            "prediction_label": prediction_label,
            "prediction": prediction,
            "accuracy": accuracy,
            "malignant": malignant,
            "saliency_map_url": saliency_map_url
        }

    else:
        print("Request failed.")
        print("Status Code:",response.status_code)
        print("Response: ", response.text)

def classify_cloud_run(image_path, metadata):
    '''
    Call to CloudRun API for prediction 
    '''
    # API Endpoint hosted local container. 
    # NOTE: for deployment, the URL needs to be changed to cloud URL API
    # e.g. "https://dermaid-prediction-qhg6oc4tja-uc.a.run.app/classify_post/"

    predict_url="http://host.docker.internal:8080/classify_post/"

    image_name = os.path.basename(image_path)
 
    files={
        'image':(image_name, open(image_path, 'rb'), 'image/jpeg')
    } 
    
    # Making a POST request to the API
    response = requests.post(predict_url, files=files, data=metadata)

    # Verifying response
    if response.status_code == 200:
        print("Request successfull.")

        response_data = response.json()

        results = response_data.get('results')
        print(results)
        return {"results":results}
    
    else:
        print("Request failed.")
        print("Status Code:",response.status_code)
        print("Response: ", response.text)

def gradcam_cloud_run(image_path, metadata):
    '''
    Call to CloudRun API for prediction 
    '''
    # API Endpoint hosted local container. 
    # NOTE: for deployment, the URL needs to be changed to cloud URL API
    # e.g. "https://dermaid-saliency-maps-qhg6oc4tja-uc.a.run.app/gradcam_post/" 

    saliency_url="http://host.docker.internal:8181/gradcam_post/"

    image_name = os.path.basename(image_path)
 
    files={
        'image':(image_name, open(image_path, 'rb'), 'image/jpeg')
    }

   
    # Making a POST request to the API
    response = requests.post(saliency_url, files=files, data=metadata)

    # Verifying response
    if response.status_code == 200:
        print("Request successfull.")

        response_data = response.json()

        # results = response_data.get('results')
        # print(results)
        return response_data
    
    else:
        print("Request failed.")
        print("Status Code:",response.status_code)
        print("Response: ", response.text)        