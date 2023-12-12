# <img style="float: left; padding-right: 10px; width: 45px" src="https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/iacs.png"> CS115: Advanced Practical Data Science 

## Final Project: DermAID - Skin Cancer Detection
### Milestone 6


**Harvard University, Spring 2023**<br/>
**Instructors**: Pavlos Protopapas<br/>
**Teaching Assistant**: Andrew Smith



**Authors:** Arash Sarmadi, Artemio Mendoza, Georg Ziegner, Sahil Sakhuja, Michael Choi

**Presentation Video:** https://www.youtube.com/watch?v=qHw_RU1XoTY <br/>
**Blog Post:** https://medium.com/institute-for-applied-computational-science/spotting-survival-689913b43d5e <br/>

<hr style="height:2pt">

Welcome to DermAID —  In today's era where healthcare meets technology, early detection of diseases can be the key to successful treatments. Skin cancer, being one of the most common types of cancer, often goes unnoticed until it reaches a critical stage. Our mission? To aid in the early detection of skin cancer by leveraging the power of deep learning.

<a id="contents"></a>
## Table of Contents

1. [Project Introduction and Organization](#organization)
2. [Final proposed solution](#solution)
    - 2.1 [Scaling using Kubernetes clusters](#scaling)
    - 2.2 [Deployment Plan](#deployment)
    - 2.3 [CI/CD using Github Actions](#ci/cd)
    - 2.4 [Miscellaneous improvements](#miscellaneous)
3. [Containers from previous milestones](#containers)
4. [References](#references)

<a id="organization"></a>
## 1. Introduction and Project Organization
### Description
DermAID is a web-based application that allows the multi-label classification of eight different types of skin lesions, ranging from benign lesions like common moles (melanocytic nevi) to malignant types like melanoma. The classification is done using a deep learning based model that was trained with a dataset of more than 25,000 images of skin lesions and corresponding metadata.

Through the primary user interface, users (patients, physicians or dermatologists) can upload their own images of skin lesions along with metadata on the patient’s gender, age and the body part on which the skin lesion occurs. Based on this data, the app will report a classification along with its probability and a saliency map offering additional clues on features in the image that led to the classification.

A secondary user interface can be accessed by medical experts (dermatologists) only and is used to validate the model’s classification of user-provided data. This allows us to detect and address possible drops in performance. At the same time, the model can be re-trained using the expert-labeled data to further improve the model's performance and to expose it to a wider variety of skin types.

![use case scenarios](/images/use_cases.jpg)


### Project Organization
<summary>Click on details to expand
</summary>
<details>


Project Organization
------------
      ├── README.md
      ├── LICENSE
      ├── .gitignore
      ├── .gitmodules
      ├── images
      │  ├── frontend_analyze.png
      │  ├── frontend_prediction.png
      │  ├── github_actions.png
      │  ├── kubernetes_cluster.png
      │  ├── kubernetes_cluster_ip.png
      │  └── use_caes.jpg
      └── src
            ├── 01_get_raw_data
            │   ├── .dockerignore
            │   ├── .gitignore
            │   ├── Dockerfile
            │   ├── docker-shell.sh
            │   ├── get_data.py
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   └── README.md
            ├── 02_preprocess_data
            │   ├── .dockerignore
            │   ├── .gitignore
            │   ├── docker-shell.bat
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── preprocess_data.py
            │   └── README.md
            ├── 03_train_model
            │   ├── utilities
            │   │ ├── __init__.py
            │   │ ├── model_compression.py
            │   │ ├── model_quantization.py
            │   │ └── utils.py
            │   ├── .dockerignore
            │   ├── .gitignore
            │   ├── docker-shell.bat
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── dvc_pull.sh
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── README.md
            │   ├── run-commands.sh
            │   ├── run-training.sh
            │   └── training.py
            ├── 04_classify_n_upload
            │   ├── data
            │   │ ├── ISIC_0026403.jpg
            │   │ ├── ISIC_0026421.jpg
            │   │ ├── ISIC_0026456.jpg
            │   │ ├── ISIC_0026531.jpg
            │   │ └── ISIC_0026546.jpg
            │   ├── .gitignore
            │   ├── docker-shell.bat
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── mock_app_prediction.py
            │   ├── README.md
            │   ├── requirements.txt
            │   └── sample_curl.sh
            ├── 05_saliency_maps
            │   ├── data
            │   │ ├── ISIC_0026403.jpg
            │   │ ├── ISIC_0026421.jpg
            │   │ ├── ISIC_0026456.jpg
            │   │ ├── ISIC_0026531.jpg
            │   │ └── ISIC_0026546.jpg
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── mock_app_saliency_maps.py
            │   ├── README.md
            │   ├── requirements.txt
            │   ├── run-server.sh
            │   └── sample_curl.sh
            ├── 06_labeling
            │   ├── docker-compose.yml
            │   ├── docker-shell.bat
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── label_interface_code.txt
            │   ├── label_studio.py
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   └── README.md
            ├── 07_frontend
            │   ├── api-service
            │   │ ├── api
            │   │ │ ├── model.py
            │   │ │ ├── service.py
            │   │ │ └── simple_call.py
            │   │ ├── docker-entrypoint.sh
            │   │ ├── docker-shell.bat
            │   │ ├── docker-shell.sh
            │   │ ├── Dockerfile
            │   │ ├── Pipfile
            │   │ └── Pipfile.lock
            │   ├── data
            │   │ ├── ISIC_0026403.jpg
            │   │ ├── ISIC_0026421.jpg
            │   │ ├── ISIC_0026456.jpg
            │   │ ├── ISIC_0026531.jpg
            │   │ └── ISIC_0026546.jpg
            │   ├── deployment
            │   │ ├── nginx-conf
            │   │ │ └── nginx
            │   │ │   └── nginx.conf
            │   │ ├── .gitignore
            │   │ ├── deploy-create-instance.yml
            │   │ ├── deploy-docker-images.yml
            │   │ ├── deploy-k8s-cluster.yml
            │   │ ├── deploy-k8s-tic-tac-toe.yml
            │   │ ├── deploy-provision-instance.yml
            │   │ ├── deploy-setup-containers.yml
            │   │ ├── deploy-setup-webserber.yml
            │   │ ├── deploy-update-k8s-cluster.yml
            │   │ ├── docker-entrypoint.sh
            │   │ ├── docker-shell.bat
            │   │ ├── docker-shell.sh
            │   │ ├── Dockerfile
            │   │ ├── gitactions-deploy-app.sh
            │   │ └── inventory.yml
            │   ├── frontend-react
            │   │ ├── images
            │   │ │ ├── app-building-crashcourse.png
            │   │ │ ├── react-01.png
            │   │ │ ├── react-02.png
            │   │ │ ├── react-03.png
            │   │ │ ├── react-04.png
            │   │ │ ├── react-05.png
            │   │ │ ├── react-06.png
            │   │ │ └── react-07.png
            │   │ ├── public
            │   │ │ ├── index.html
            │   │ │ └── manifest.json
            │   │ ├── src
            │   │ │ ├── app
            │   │ │ │  ├── App.css
            │   │ │ │  ├── App.js
            │   │ │ │  ├── AppRoutes.js
            │   │ │ │  └── Theme.js
            │   │ │ ├── common
            │   │ │ │  ├── Content
            │   │ │ │  │ ├── index.js
            │   │ │ │  │ └── styles.js
            │   │ │ │  ├── Footer
            │   │ │ │  │ ├── index.js
            │   │ │ │  │ └── styles.js
            │   │ │ │  └── Header
            │   │ │ │    ├── index.js
            │   │ │ │    └── styles.js
            │   │ │ ├── components
            │   │ │ │  ├── Classify
            │   │ │ │  │ ├── index.js
            │   │ │ │  │ └── styles.js
            │   │ │ │  ├── Error
            │   │ │ │  │ └── 404.js
            │   │ │ │  ├── Home
            │   │ │ │  │ ├── index.js
            │   │ │ │  │ └── styles.js
            │   │ │ │  └── TOC
            │   │ │ │    ├── index.js
            │   │ │ │    └── styles.js
            │   │ │ ├── services
            │   │ │ │  ├── Common.js
            │   │ │ │  └── DataService.js
            │   │ │ ├── index.css
            │   │ │ └── index.js
            │   │ ├── .dockerignore
            │   │ ├── .env.development
            │   │ ├── .env.production
            │   │ ├── .gitignore
            │   │ ├── docker-shell.bat
            │   │ ├── docker-shell.sh
            │   │ ├── Dockerfile
            │   │ ├── Dockerfile.dev
            │   │ ├── package.json
            │   │ └── yarn.lock
            │   ├── labeling
            │   │ ├── docker-volumes
            │   │ │ ├── label-studio
            │   │ │ │  ├── export
            │   │ │ │  │ ├── project-1-at-2023-11-19-08-22-cc80338d.json
            │   │ │ │  │ └── project-1-at-2023-11-19-08-22-cc80338d-info.json
            │   │ │ │  ├── media
            │   │ │ │  │ └── upload
            │   │ │ │  │   └── 2
            │   │ │ │  │     ├── 3cbebb7e-image_test.json
            │   │ │ │  │     └── caa08a73-image_test.json
            │   │ │ │  └── label_studio.sqlite3
            │   │ ├── Dockerfile
            │   │ ├── image_test.json
            │   │ └── label_interface_code.txt
            │   └── README.md
            ├── 08_versioning
            │   ├── .dvc
            │   │ ├── .gitignore
            │   │ └── config
            │   ├── .dvcignore
            │   ├── .gitignore
            │   ├── Check_Version_Download.ipynb
            │   ├── cli.py
            │   ├── cli_old.py
            │   ├── docker-shell.sh
            │   ├── Dockerfile
            │   ├── dvc_push.sh
            │   ├── Pipfile
            │   ├── Pipfile.lock
            │   ├── README.md
            │   ├── run-commands.sh
            │   └── Versioning.dvc
            └── 09_run_kfp
                ├── .dockerinore
                ├── .gitignore
                ├── docker-entrypoint.sh
                ├── docker-shell.bat
                ├── docker-shell.sh
                ├── Dockerfile
                ├── Pipfile
                ├── Pipfile.lock
                ├── README.md
                └── run-kfp.py
</details>



<a id="solution"></a>
## 2. Proposed Solution
[Return to Table of Contents](#contents)`

In the previous milestones we  developed our solution in a series of cumulative steps, including:

    - created data pipelines, 
    - trained our predictive model
    - deployed the model to GCP
    - created a backend API server to access the deployed model
    - developed a frontend using React
    - automatic deployment to GCP using Ansible

The final steps to make our application available and deployable were to ensure scalability using Kubernetes clusters and to develop a deployment plan, incl. CI/CD capabilities.

In the following sections we describe the steps to implement the application, either in GCP or locally (option listed in Container *07_frontend/deployment*).

<a id="scaling"></a>
### 2.1. Scaling using Kubernetes clusters
[Return to Table of Contents](#contents)

To optimize accessibility and performance, our model uses Kubernetes clusters to deploy the API backend and our WebApp frontend.

The architecture focuses on scalability, enabling handling of increased requests without performance loss. It also incorporates failover mechanisms for continuous operation and load balancing to evenly distribute computational load, preventing bottlenecks. 

<a id="deployment"></a>
### 2.2. Deployment Plan
[Return to Table of Contents](#contents)

The deployment is managed from within a docker container. **With the help of Ansible Playbook scripts**, we can manage creating and updating the K8 clusters.

In the following section we list the steps to deploy DermAID to a K8s cluster on GCP. For instructions on how to deploy locally, click [here](src/07_frontend/README.md).

<a id="step1"></a>

#### Step 1 - Enable APIs in GCP
As a first step, make sure all required APIs are enabled in GCP. To do that, search for each of the following APIs in the GCP search bar, click the API to open a page with more information about the API, and then click **Enable**.

* Compute Engine API
* Service Usage API
* Cloud Resource Manager API
* Google Container Registry API
* Kubernetes Engine API

<a id="step2"></a>

#### Step 2 - Start Deployment Docker Container

Start the Docker container with the image for creating the K8 clusters. To do that:

-  change directory to **deployment**

    ```bash
    cd deployment
    ```
- start docker container   
    Unix:
    ```bash
    sh docker-shell.sh
    ```
    
    Windows:

    ```batch
    docker-shell.bat
    ```

- Check the tool versions
    ```bash 
    gcloud --version
    kubectl version
    kubectl version --client
    ```

- Ensure you are authenticated to GCP
    ```bash
    gcloud auth list
    ```

<a id="step3"></a>

#### Step 3 - Build & Deploy Containers

*Not required if already done.*

The relevant containers for the frontend (User frontend based on React & label studio) and the backend API service must be built and deployed to Google Container Registry before deploying a full Kubernetes cluster.

This can be simply achieved by running the following command.

```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```


<a id="step4"></a>

#### Step 4 - Create & Deploy Cluster

Run the following command ro create and deploy your K8 cluster:

```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
```
Below we can observe our K8 cluster named *dermaid-app-cluster* created in GPC, as well as the external EndPoint IP

![k8 cluster](/images/kubernetes_cluster.png)
![kubernetes ingress IP](/images/kubernetes_cluster_ip.png)


<a id="step5"></a>

#### Step 5 - View and access the application
* Copy the `nginx_ingress_ip` that will be displayed after successfully creating the K8 cluster.
* Open `http://<YOUR INGRESS IP>.sslip.io` in your web browser.

        **Note**: Using sslip.io in conjunction with your ingress IP address in Kubernetes on GCP (Google Cloud Platform) is a method to simplify the process of mapping a public IP address to a domain name, particularly when you don't own a domain or don't want to configure DNS settings for a small project or a temporary environment.

Now we can observe the landing page of the application deployed on GCP. We can then analize a sample from the image database...

![webapp analizing](/images/frontend_analyze.png)

...and obtain a prediction.

![webapp prediction](/images/frontend_prediction.png)

<a id="step6"></a>

#### Step 6 - View and access Label Studio application
* Copy the `nginx_ingress_ip` that will be displayed after successfully creating the K8 cluster.
* Open `http://<YOUR INGRESS IP>.sslip.io/labeling/` in your web browser.


<a id="step7"></a>

#### Step 7 - Delete Cluster

To delete the K8 cluster, run:
```bash
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=absent
```

<a id="ci/cd"></a>
### 2.3. CI/CD using Github Actions
[Return to Table of Contents](#contents)

After deploying a Kubernetes cluster, our last task was to implement CI/CD automation through Github Actions. 

We created a separate Kubernetes update script which is triggered by Github Actions when the commit message contains “/run-app-deploy". This automatically rebuilds the 3 containers (frontend, api-service and labeling), publishes them to Google Container Registry and updates the deployment of these containers on the Kubernetes Cluster.

![github actions](/images/github_actions.png)

There are 2 actions taken in the CI/CD pipeline:
1. Build and deploy new containers with any code changes that have been made in the application - this is done using the same ansible command as above for deploying the containers.
1. Update the K8s cluster on GCP - this is done using a different ansible yml file (deploy-update-k8s-cluster.yml) which only updates the containers on the cluster and does not change any of the cluster setup.

<a id="miscellaneous"></a>
### 2.4 Miscellaneous improvements
[Return to Table of Contents](#contents)

For our final version, we further improved the primary user interface to:

- allow users to enter relevant metadata for the prediction
- return saliency map of the image classification
- include a link for the medical experts to review the classification 


---
<a id="containers"></a>
<summary>

## 3. Containers from previous Milestones
  
[Return to Table of Contents](#contents)

Note: For better readability of this file, we will only give a short overview of each container below. More details on each container can be found in dedicated README files for each container.


> **Note**: The list of containers is collapsed, please, click on <b>Details</b> to expand it.
</summary>
<details> 

---

**Container 01: Data Download**

This container is the first part of our data pipeline. It downloads the zipped images from the ISIC 2019 Challenge website, extracts images and uploads them to GCS. It also downloads metadata and labels and stores them in a GCS bucket.

[More information](src/01_get_raw_data/README.md)

---

**Container 02: Data Preprocessing**

This container is the second part of our data pipeline. It collects the raw data from GCS, converts it into tf Datasets and then stores the complete datasets as tfrecords, back into GCS.

[More information](src/02_preprocess_data/README.md)

---

**Container 03: Model Training**

This container performs the training of our models. It can run both locally and serverless on VertexAI. It collects the tfrecords datasets stored on GCP. In order to pull the correct data, the container requires first to download the versioned pickle file containing the paths to the tfrecords files to be used for training (based on the version tag supplied as a command line argument when creating the vertex AI pipeline), and then creates the model and performs the training.

The container supports multi-GPU training via a configurable parameter. It also has experiment tracking using Weights & Biases. Optional model compression methods are also implemented.

[More information](src/03_train_model/README.md)

---

**Container 04: Inference API**

This container uses a trained model from GCS produced by container 03 to classify new images.
FastAPI is implemented to expose RESTful API for classification, to be used by the front-end application to upload image and metadata for classification. It will also upload the metadata, the raw and preprocessed new image to GCS afterwards.
A CLI is also included to perform the same functionalities.
These data will be used by containers 5 and 6 for Saliency Maps and Labelling.

[More information](src/04_classify_n_upload/README.md)

---

**Container 05: Saliency Maps API**

This container exposes an API that can be called from the frontend application to upload an image and retrieves the model prediction and the original image with a saliency map overlay showcasing the significant areas of the image which impacted the prediction.

[More information](src/05_saliency_maps/README.md)

---

**Container 06: Label Studio**

This container handles our app's second front-end for our other users i.e. Dermatologists, using Label Studio. Dermatologists will be able to use Label Studio to assign the correct labels to different images uploaded by the actual end users of our web app.
Iput to the container are unlabeled patient images and environment variables for Google Cloud Services. 
Output from the container are labeled patient images stored in Google Cloud Storage.

[More information](src/06_labeling/README.md)

---

**Container 07: FrontEnd**

This container handles the frontend, the api-services, and the ansible playbooks for deployment.

The frontend was is based on reac framework, and the api-services are implemented using FastAPI, as a wrapper for the CloudRun end-points.

The deployment to GCP is performed using Ansible Playbooks in automated fashion. It includes the following steps:

* Build and push Docker images with code to the Google Container Registry (GCR)
* Provision Virtual Machine Instance and install all dependencies
* Setup Docker Images from GCR into the VM instance
* Setup Nginx webserver as a reverse proxy

Click here for [detailed information on how to deploy the app](src/07_frontend/README.md) on how to deploy the application.


[More information](src/07_frontend/README.md)

---

**Container 08: Versioning**

This is a separate git project to track and version the datasets in the DermAID App.
We will be using data versioning for fine-tuning the model based on labeling done by Dermatologists. The end result of the pre-processing of data in Container 2 is the creation of tfrecords on GCP. At the end of the pre-processing, the paths created by Container 2 are stored in a temp file on GCP. This container is the next step in the Vertex Pipeline - it downloads the temporary file created by Container 2 and versions it using the tag supplied as a CLI argument when creating the Vertex pipeline.
Input to the container are: GCP credentials, GCP bucket & project details, Git credentials and tag.
Output from the container is an updated DVC file which is commited to the Git repo with the specified commit message and tag.

[More information](https://github.com/csci-e-115/dermaid-versioning/tree/milestone4)

---
**Container 09: Vertex AI Pipeline Runner**

This container constructs the Kubeflow Pipeline and submit to Vertex AI to run. It will compile the defined pipeline and container configuration into pipeline.yaml for job submission.
It assumes containers it used in the pipeline to have already been uploaded to a container repo such as dockerhub. In this case they are:
  - container 1: crispinnosidam/01-get-raw-data
  - container 2: crispinnosidam/02-preprocess-data
  - container 3: sahilsakhuja/dermaid-versioning
  - container 4: crispinnosidam/03-train-model

[More information](src/09_run_kfp/README.md)
</details>

<a id="references"></a>
## 4. References
[Return to Table of Contents](#contents)

1. : BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona  

2. : HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; https://doi.org/10.1038/sdata.2018.161  

3. : MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368  
