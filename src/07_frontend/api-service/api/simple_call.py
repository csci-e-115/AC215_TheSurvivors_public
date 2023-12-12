import requests
import json
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

# API Endpoint
predict_url="https://dermaid-prediction-qhg6oc4tja-uc.a.run.app/classify_post/"
saliency_url="https://dermaid-saliency-maps-qhg6oc4tja-uc.a.run.app/gradcam_post/"
saliency_url="https://localhost:8181/gradcam_post/"

image_name='ISIC_0026403.jpg'
image_path = f'../../data/{image_name}'

files={
    'image':(image_name, open(image_path, 'rb'), 'image/jpeg')
}

metadata = {
    'age': 60,
    'location': 'head/neck',
    'sex':'male',
}


# Making a POST request to the API
response = requests.post(saliency_url, files=files, data=metadata)

# Verifying response
if response.status_code == 200:
    print(f"Request successfull. {response}")

    # # response_data = json.loads(response.json().replace("'", "\""))
    # response_data = response.json()

    # prediction_label = response_data.get('prediction_name')
    # prediction = response_data.get('prediction_proba')
    # malignant = False
    # if prediction_label in malignant_lesion:
    #     malignant = True
    
    # print(f'Image [{image_name}], Diagnostic: [{prediction_label}] with probability: [{prediction}], malignant: {malignant}')

else:
    print("Request failed.")
    print("Status Code:",response.status_code)
    print("Response: ", response.text)