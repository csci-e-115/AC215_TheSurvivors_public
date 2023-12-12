from fastapi import FastAPI,  Form, File
from starlette.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import os
from fastapi import File
from tempfile import TemporaryDirectory
from api import model


# Setup FastAPI app
app = FastAPI(title="DermAID API Server", description="Prediction API Server", version="v0.3")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to the API Service"}


@app.post("/predict")
async def predict(file: bytes = File(...),
                  age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                  location: str = Form(..., title="Location", description="Specify location"),
                  sex: str = Form(..., title="Sex", description="Specify sex")):
    
    # Form Metadata for prediction
    metadata = {
     'age': age,
     'location': location,
     'sex':sex,
    }

    # prediction input
    print("predict file:", len(file), type(file))
    print(f'Metadata: {metadata}')

    cloud_run = True

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)      

        # Make prediction
        prediction_results = {}
        if cloud_run:
            prediction_results = model.make_prediction_cloud_run(image_path, metadata)
        else:
            prediction_results = model.make_prediction(image_path, metadata)
            

    print(prediction_results)
    return prediction_results

@app.post("/classify_post")
async def DermAIDClassificationPredict(file: bytes = File(...),
                  age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                  location: str = Form(..., title="Location", description="Specify location"),
                  sex: str = Form(..., title="Sex", description="Specify sex")):
    
    # Form Metadata for prediction
    print('classify post')
    metadata = {
     'age': age,
     'location': location,
     'sex':sex,
    }

    # prediction input
    print("predict file:", len(file), type(file))
    print(f'Metadata: {metadata}')

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)      

        # Make prediction
        prediction_results = {}
        prediction_results = model.classify_cloud_run(image_path, metadata)
            

    print(prediction_results)
    return prediction_results

@app.post("/gradcam_post")
async def gradcam_post(file: bytes = File(...),
                  age: int = Form(..., title="Age", description="Specify age", ge=0, le=85),
                  location: str = Form(..., title="Location", description="Specify location"),
                  sex: str = Form(..., title="Sex", description="Specify sex")):
    
    # Form Metadata for prediction
    print('gradcam post v2')    
    metadata = {
     'age': age,
     'location': location,
     'sex':sex,
    }

    # prediction input
    print("predict file:", len(file), type(file))
    print(f'Metadata: {metadata}')

    # Save the image
    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, "test.png")
        with open(image_path, "wb") as output:
            output.write(file)      

        # Make prediction
        prediction_results = {}
        prediction_results = model.gradcam_cloud_run(image_path, metadata)
            

    print(prediction_results)
    return prediction_results