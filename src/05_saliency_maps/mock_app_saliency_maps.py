import io, uuid, os.path, json
import datetime

from google.cloud import storage
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tf_keras_vis.gradcam import Gradcam

# Needed for the score function
from tf_keras_vis.utils.scores import CategoricalScore

from typing import Annotated
from fastapi import FastAPI, File, UploadFile, Form
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load environment variables
gcp_project = os.environ["GCP_PROJECT"]
bucket_name = os.environ["GCS_BUCKET_NAME"]
models = os.environ["GCS_MODELS"]
GCS_SALIENCY_MAPS = os.environ["GCS_SALIENCY_MAPS"]
GCS_IMAGES_UPLOAD = os.environ["GCS_IMAGES_UPLOAD"]
GCS_JSON_UPLOAD = os.environ["GCS_JSON_UPLOAD"]

model_dir = f"gs://{bucket_name}/{models}/"
img_gcp_url = "https://storage.googleapis.com/{}/{}"


opts_label_codes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
opts_label_names = [
    "Melanoma","MelanocyticNevus","BasalCellCarcinoma","ActinicKeratosis",
    "BeningKeratosis","Dermatofibroma","VascularLesion","SquamousCellCarcinoma"
    ]

opts_anatomic_site = [
    'anterior torso', 'head/neck', 'lateral torso', 'lower extremity',
    'oral/genital', 'palms/soles', 'posterior torso', 'upper extremity'
    ]

# Initiate Storage client -
storage_client = storage.Client(project=gcp_project)
# Get reference to bucket
bucket = storage_client.bucket(bucket_name)

opts_age = list(range(0, 86, 5))
opts_sex = ['female', 'male']

# This layer gives better explanation than "conv5_block1_0_relu" and penultimate
inspected_layer_name = "conv5_block16_2_conv"

trained_model = tf.keras.models.load_model(model_dir)
trained_layer=trained_model.get_layer(inspected_layer_name).output

def get_gradcam(image, trained_model, trained_layer, inspected_layer_name, score):
    # Used to dived by "zero"
    epsilon = 1e-8

    # Create a new model for visualization   
    trained_model_input=trained_model.input[0]

    visualization_model = Model(inputs=trained_model_input,
                                outputs=trained_layer)

    gradcam_visualization = Gradcam(visualization_model)

    # cam = gradcam_visualization(score, image[np.newaxis, ...], penultimate_layer=inspected_layer_name)
    cam = gradcam_visualization(score, image, penultimate_layer=inspected_layer_name)

    # Normalization using epsilon to avoid division by zero
    heatmap = (cam - cam.min()) / (cam.max() - cam.min() + epsilon)

    return heatmap

def get_gradcam_img(image, trained_model, trained_layer, inspected_layer_name, score, random_id, my_cmap='viridis'):
    # get heatmap

    heatmap = get_gradcam(image, trained_model, trained_layer, inspected_layer_name, score)

    # plt.figure(figsize=(4,4))
    plt.imshow(np.squeeze(image, axis=0), cmap='gray')
    plt.imshow(heatmap[0], cmap=my_cmap, alpha=0.4) # Overlaying the heatmap on the image
    plt.axis('off')
    
    # Save the figure to an in-memory binary stream
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0) 
    buf.seek(0)

    # Load the image from the binary stream into a PIL Image object
    image_in_memory = Image.open(buf)

    # save image to a bucket    
    storage_client = storage.Client()
    bucket=storage_client.bucket(bucket_name)
    blob = bucket.blob(f'{GCS_SALIENCY_MAPS}/{random_id}.{image_in_memory.format}')
    buf.seek(0)

    blob.upload_from_file(buf, content_type="image/jpeg")

    plt.close()
    
    return image_in_memory

def get_one_hot(targets, nb_classes):
    targets = np.array([[targets]]).reshape(-1)
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

# Entry point for command line (e.g. curl)
@app.post("/gradcam_post/")
async def classify_post(image: UploadFile = File(...), 
                   age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                   location: str = Form(..., title="Location", description="Specify location"),
                   sex: str = Form(..., title="Sex", description="Specify sex")):

# generate random id for this transaction
    random_id = str(uuid.uuid4())
    
    img_ext = image.filename.split('.')[-1].lower()
    img_content_type = image.content_type

    image = Image.open(image.file)
    image = image.resize((256, 256))
    image_obj = np.array(image)
    image_obj = image_obj / 256.0

    image_obj = np.expand_dims(image_obj, axis=0)

    age_idx = opts_age.index(age)
    age_oh = get_one_hot(age_idx, len(opts_age))

    site_idx = opts_anatomic_site.index(location)
    site_oh = get_one_hot(site_idx, len(opts_anatomic_site))

    sex_idx = opts_sex.index(sex)
    sex_oh = get_one_hot(sex_idx, len(opts_sex))

    metadata = np.concatenate((age_oh, site_oh, sex_oh), axis=1)

    predictions = trained_model.predict((image_obj, metadata))
    max_prob_idx = np.argmax(predictions)
    max_proba = np.max(predictions)

    score = CategoricalScore([max_prob_idx])

    img_gradcam = get_gradcam_img(image_obj, trained_model, trained_layer, inspected_layer_name, score, random_id)

    # save the image to gcp
    storage_client = storage.Client()
    bucket=storage_client.bucket(bucket_name)
    blob = bucket.blob(f'{GCS_IMAGES_UPLOAD}/{random_id}.{img_ext}')

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="jpeg")
    # img_byte_arr = img_byte_arr.getvalue()
    img_byte_arr.seek(0)

    blob.upload_from_file(img_byte_arr, content_type=img_content_type)

    # create a json task for label studio
    task = {
        "data": {
            "image_url": f"gs://{bucket_name}/{GCS_IMAGES_UPLOAD}/{random_id}.{img_ext}",
            "user_age": age,
            "user_gender": sex,
            "user_anatom_site_general": location,
            "ref_id": random_id,
            "meta_info": {
                "timestamp": str(datetime.datetime.now())
                }
            }
        }
    
    # upload task into a json file
    blob = bucket.blob(f'{GCS_JSON_UPLOAD}/{random_id}.json')
    blob.upload_from_string(json.dumps(task))

    print(f'Predictions [{predictions}]')

    return {
        "identifier": random_id,
        "prediction_idx": int(max_prob_idx),
        "prediction_proba": float(max_proba),
        "prediction_code": opts_label_codes[max_prob_idx],
        "prediction_name": opts_label_names[max_prob_idx],
        "saliency_map_url": img_gcp_url.format(bucket_name, f'{GCS_SALIENCY_MAPS}/{random_id}.{img_gradcam.format}')
    }