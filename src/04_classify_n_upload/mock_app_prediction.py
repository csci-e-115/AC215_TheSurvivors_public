import os, io
import os.path
from google.cloud import storage
from fastapi import FastAPI, UploadFile, Form, Query, Body, File
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
from zipfile import ZipFile
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np

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
bucket_name = os.environ["GCS_BUCKET_NAME"]
gcp_project = os.environ["GCP_PROJECT"]
model_dir = os.environ["GCS_MODELS"]
unlabelled_raw_imgs_dir = os.environ["RAW_IMAGES_PATH"]
unlabelled_metadata_dir = os.environ["METADATA_PATH"]
unlabelled_preproc_imgs_dir = os.environ["PREPROC_IMAGES_PATH"]

label_codes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

anatomic_site = ['anterior torso', 'head/neck', 'lateral torso', 'lower extremity',
                 'oral/genital', 'palms/soles', 'posterior torso', 'upper extremity']

raw_metadata_cols = ["image", "age_approx", "anatom_site_general", "lesion_id", "sex"]

metadata_cols = ['age_approx_0.0', 'age_approx_10.0','age_approx_15.0', 'age_approx_20.0',
                 'age_approx_25.0', 'age_approx_30.0', 'age_approx_35.0', 'age_approx_40.0',
                 'age_approx_45.0', 'age_approx_5.0', 'age_approx_50.0', 'age_approx_55.0',
                 'age_approx_60.0', 'age_approx_65.0', 'age_approx_70.0',
                 'age_approx_75.0', 'age_approx_80.0', 'age_approx_85.0',
                 'anatom_site_general_anterior torso', 'anatom_site_general_head/neck',
                 'anatom_site_general_lateral torso', 'anatom_site_general_lower extremity',
                 'anatom_site_general_oral/genital', 'anatom_site_general_palms/soles',
                 'anatom_site_general_posterior torso',
                 'anatom_site_general_upper extremity',
                 'sex_female', 'sex_male']

# Initiate Storage client -
storage_client = storage.Client(project=gcp_project)
# Get reference to bucket
bucket = storage_client.bucket(bucket_name)

def download_bucket_file(source_fp, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    with ZipFile(io.BytesIO(storage.Blob(source_fp, bucket).download_as_bytes())) as zip_file:
        zip_file.extractall((dest_folder))


def download_bucket_dir(source_dir):
    for blob in storage_client.list_blobs(bucket_name, prefix=source_dir):
        #print(blob.name)
        if blob.name[-1] != "/":
            blob.download_to_filename(blob.name)
        else:
            if not os.path.exists(blob.name):
                os.makedirs(blob.name)


def upload_file(source_fp, dest_folder):
    # Destination path in GCS
    get_file_name = lambda path: os.path.split(path)[1]
    fname = get_file_name(source_fp)
    destination_blob_name = f"{dest_folder}/{fname}"
    dst_blob = bucket.blob(destination_blob_name)
    if not dst_blob.exists():
        print(f"Uploading {fname} to {dest_folder}")
        dst_blob.upload_from_filename(source_fp)

def create_image_tensor(image):
    with TemporaryDirectory() as tempdir:
        shutil.copy(image, tempdir)
        # print(os.listdir(tempdir), tempdir)

        image_ds = tf.keras.utils.image_dataset_from_directory(
            tempdir,
            labels=None,
            batch_size=1,
            image_size=(256, 256),
            shuffle=False
        )

        rescaled_ds = image_ds.map(
            lambda x: tf.keras.Sequential([ tf.keras.layers.Rescaling(1 / 255) ])(x))

        assert(len(rescaled_ds) == 1)
        return iter(rescaled_ds).next()


def save_image_from_tensor(img_tensor, img_path):
    Image.fromarray((img_tensor.numpy()[0] * 255).astype(np.uint8),
                    "RGB").save(img_path)


def create_metadata_tensor(metadata):
    # the input should already be of type int
    metadata_tensor = tf.convert_to_tensor(metadata.to_numpy().astype("int32"))
    return tf.reshape(metadata_tensor, [-1, 1, metadata.shape[1]])[0]


def classify(model_dir, img_tensor, metadata_tensor):
    model = tf.keras.models.load_model(model_dir)
    return model.predict((img_tensor, metadata_tensor))[0]


def process_image(image, age=None, location=None, sex=None, load_model=False):   
    '''
    Stitches together image with metadata and creates a TF
    '''
    print(f' Process Image input: age [{age}], location [{location}], sex [{sex}] ')

    if not os.path.isfile(image):
        raise ValueError(f"Invalid file name: {image} is not a file or does not exist.")
    
    if os.path.splitext(image)[-1] not in [".bmp", ".jpg", ".jpeg", ".png", ".gif"]:
       raise ValueError("Invalid file type: must be in one of these extensions {'bmp', 'jpg', 'jpeg', 'png', 'gif'}")

    if age is not None and age not in range(0,90,5):
        raise ValueError("Invalid age group: must be a multiple of 5 between 0 to 85.")

    if location is not None and location not in anatomic_site:
        raise ValueError(f"Invalid anatomic site: must be one of {anatomic_site}.")

    if sex is not None and sex not in ["male", "female"]:
        raise ValueError("Invalid sex: must be one of {male, female}.")
    
    # upload raw image to bucket
    upload_file(image, unlabelled_raw_imgs_dir)

    # convert image, create image tensor
    image_tensor=create_image_tensor(image)

    # save resized image and upload to bucket
    if not os.path.exists(unlabelled_preproc_imgs_dir):
          os.makedirs(unlabelled_preproc_imgs_dir)
         
    img_bname = os.path.basename(image)
    save_image_from_tensor(image_tensor, f"{unlabelled_preproc_imgs_dir}/{img_bname}")
    upload_file(f"{unlabelled_preproc_imgs_dir}/{img_bname}", unlabelled_preproc_imgs_dir)

    # build dataframe to save to csv and upload
    img_name=os.path.splitext(img_bname)[0]
    raw_meta = pd.DataFrame(dict(zip(raw_metadata_cols,
                                     [[img_name],
                                      ["" if age is None else age],
                                      ["" if location is None else location],
                                      [""],
                                      ["" if sex is None else sex]])))
    print(f'Raw Metadata: {raw_meta}')
    raw_meta_fname=f"{unlabelled_metadata_dir}/{img_name}.csv"
    if not os.path.exists(unlabelled_metadata_dir):
        os.makedirs(unlabelled_metadata_dir)
    raw_meta.to_csv(raw_meta_fname, index=False)
    upload_file(raw_meta_fname, unlabelled_metadata_dir)

    # build dataframe to create metadata tensor
    onehot_meta = pd.DataFrame(dict(zip(metadata_cols, [[0]] * len(metadata_cols))))
    if age:
        onehot_meta[f"age_approx_{str(age) + '.0'}"] = 1
    if sex:
        onehot_meta[f"sex_{sex}"] = 1
    if location:
        onehot_meta[f"anatom_site_general_{location}"] = 1

    # create metadata tensor
    metadata_tensor=create_metadata_tensor(onehot_meta)

    # download model for inference and classify
    if not os.path.exists(model_dir) or load_model:
        print("Downloading Model...")
        download_bucket_dir(model_dir)

    model = tf.keras.models.load_model(model_dir)
    results = model.predict((image_tensor, metadata_tensor))[0]

    pred = label_codes[np.argmax(results[0])]

    # convert numpy.float32 back to float otherwise cannot be results cannot be passed to Response
    pred_proba = dict(zip(label_codes, [val.item() for val in results]))

    print(f"Classification Result: Class {pred}")
    print(f"Probability of Classification {pred_proba}")

    return {"pred": pred, "pred_proba": pred_proba}


# Entry point for Web
@app.get("/classify/")
async def classify(image: UploadFile = File(...), 
                   age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                   location: str = Form(..., title="Location", description="Specify location"),
                   sex: str = Form(..., title="Sex", description="Specify sex")):
    
     # Here, we are expecting the parameters directly from the URL's query parameters.
    local_raw_img_path = Path(f"temp_{unlabelled_raw_imgs_dir}")
    full_img_path = Path(f"{local_raw_img_path}/{image.filename}")


    if not os.path.isdir(local_raw_img_path):
        os.makedirs(local_raw_img_path)
    with full_img_path.open('wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    return process_image(full_img_path, age, location, sex, load_model=False)

# Entry point for command line (e.g. curl)
@app.post("/classify_post/")
async def classify_post(image: UploadFile = File(...), 
                   age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                   location: str = Form(..., title="Location", description="Specify location"),
                   sex: str = Form(..., title="Sex", description="Specify sex")):

    # Here, we are expecting the parameters directly from the URL's query parameters.
    local_raw_img_path = Path(f"temp_{unlabelled_raw_imgs_dir}")
    full_img_path = Path(f"{local_raw_img_path}/{image.filename}")

    if not os.path.isdir(local_raw_img_path):
        os.makedirs(local_raw_img_path)

    with full_img_path.open('wb') as buffer:
        shutil.copyfileobj(image.file, buffer)

    label_names = {
        "MEL": "Melanoma",
        "NV": "Melanocytic nevus",
        "BCC": "Basal cell carcinoma",
        "AK": "Actinic keratosis",
        "BKL": "Benign keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular lesion",
        "SCC": "Squamous cell carcinoma"}

    malignant_lesion = ["MEL", "BCC", "AK", "SCC"]
    results = process_image(full_img_path, age, location, sex, load_model=False)

    results["pred_proba"] = {k: round(v, 6) for k, v in results["pred_proba"].items()}
    maxkey = max(results["pred_proba"], key=results["pred_proba"].get)
    results["pred"] = [label_names[maxkey], results["pred_proba"][maxkey],
                       "Malignant" if maxkey in malignant_lesion else "Benign"]
    results["pred_proba"] = [{"name": label_names[key], "proba": val} for key, val in sorted(
        results["pred_proba"].items(),
        key=lambda item:item[1],
        reverse=True
    )]

    results = {"results": [results]}
    print(results)
    return results