# Dermaid App - Deployment & Scaling



<a id="contents"></a>
## Table of Contents

1. [Before deploying](#before)
2. [How to deploy to GCP](#deployment)
3. [Deployment with Scaling using Kubernetes](#kubernetes)  
   3.1. [Step 1: enable GCP APIs](#step1)  
   3.2. [Step 2: create local container](#step2)  
   3.3. [Step 3: push microservices into GCP registry](#step3)  
   3.4. [Step 4: create and deploy K8 Cluster](#step2)  
4. [File Structure and dependencies](#dependencies)
5. [Prerequisites](#prerequisites)
6. [Challenges and Solutions](#challenges)

<a id="description"></a>
## Description

This container handles the frontend, the api-services, the ansible playbooks for deployment, and Kubernete clusters for Scaling and balancing.

The frontend is based on the react framework, and the api-services are implemented using FastAPI, as a wrapper for the CloudRun end-points.

The deployment to GCP is performed using **Ansible Playbooks** in automated fashion. It includes the following steps:

* Build and push Docker images with code to the Google Container Registry (GCR)
* Provision Virtual Machine Instance and install all dependencies
* Setup Docker Images from GCR into the VM instance
* Setup Nginx webserver as a reverse proxy

Also, to optimize accessibility and performance, our model uses **Kubernetes clusters**. The architecture focuses on scalability, enabling handling of increased requests without performance loss. It also incorporates failover mechanisms for continuous operation and load balancing to evenly distribute computational load, preventing bottlenecks.

The deployment is managed from a docker container, with the help of Ansible Playbooks.

<a id="before"></a>
## 1. Before deploying
[Return to Table of Contents](#contents)

Before deploying, ensure you have all your container build and works locally


#### api-service
* Go to `http://localhost:9000/docs` and make sure you can see the API Docs
#### frontend-react
* Go to `http://localhost:3000/` and make sure you can see the prediction page


<a id="deployment"></a>
## 2. How to deploy to GCP 
[Return to Table of Contents](#contents)

In this section we will deploy the Dermaid App to GCP using Ansible Playbooks in a automated fashion.

### 2.1 Prerequisites 

#### 2.1.1. Enable following GCP API's
Search for each of the following APIs in the GCP search bar and click enable to activate them:

* Compute Engine API
* Service Usage API
* Cloud Resource Manager API
* Google Container Registry API

#### 2.1.2. Setup GCP Service Account for deployment.
Steps:
- To setup a service account you will need to go to [GCP Console](https://console.cloud.google.com/home/dashboard), search for  "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "deployment". 
- Give the following roles:
- For `deployment`:
    - Compute Admin
    - Compute OS Login
    - Container Registry Service Agent
    - Kubernetes Engine Admin
    - Service Account User
    - Storage Admin
- Then click done.
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Create key". A prompt for Create private key for "deployment" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the **secrets** folder.
- Rename the json key file to `deployment.json`
- Follow the same process Create another service account called `gcp-service`
- For `gcp-service` give the following roles:
    - Storage Object Viewer
- Then click done.
- This will create a service account
- On the right "Actions" column click the vertical ... and select "Create key". A prompt for Create private key for "gcp-service" will appear select "JSON" and click create. This will download a Private key json file to your computer. Copy this json file into the **secrets** folder.
- Rename the json key file to `gcp-service.json`

#### 2.1.3. Setup Docker Container (Ansible, Docker, Kubernetes)

Rather than installing all the tools for each deployment, we will use Docker to build and run a standard container with all required software.

#### Run `deployment` container
- cd into `deployment`
- Go into `docker-shell.sh` or `docker-shell.bat` and change `GCP_PROJECT` to your project id
- Run `sh docker-shell.sh` or `docker-shell.bat` for windows

- Check versions of tools:
```
gcloud --version
ansible --version
kubectl version --client
```

- Check to make sure you are authenticated to GCP
- Run `gcloud auth list`

Now you have a Docker container that connects to your GCP and can create VMs, deploy containers all from the command line


#### 2.1.4. SSH Setup
#### Configuring OS Login for service account
```
gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE
```
example: 
```
gcloud compute project-info add-metadata --project ac215-project --metadata enable-oslogin=TRUE
```

#### Create SSH key for service account
```
cd /secrets
ssh-keygen -f ssh-key-deployment
cd /app
```

#### 2.1.5. Providing public SSH keys to instances
```
gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub
```
From the output of the above command keep note of the username. Here is a snippet of the output 
```
- accountId: ac215-project
    gid: '3906553998'
    homeDirectory: /home/sa_100110341521630214262
    name: users/deployment@ac215-project.iam.gserviceaccount.com/projects/ac215-project
    operatingSystemType: LINUX
    primary: true
    uid: '3906553998'
    username: sa_100110341521630214262
```
The username is `sa_100110341521630214262`

#### 2.1.6. Deployment Setup
* Add ansible user details in inventory.yml file
* GCP project details in inventory.yml file
* GCP Compute instance details in inventory.yml file


### 2.2. Deployment

#### Build and Push Docker Containers to GCR (Google Container Registry)
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

#### Create Compute Instance (VM) Server in GCP
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```

Once the command runs successfully, get the IP address of the compute instance from GCP Console and update the appserver>hosts in inventory.yml file

#### Provision Compute Instance in GCP
Install and setup all the required components for deployment.
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

#### Setup Docker Containers in the  Compute Instance
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```


You can SSH into the server from the GCP console and see status of containers
```
sudo docker container ls
sudo docker container logs api-service -f
```

To get into a container run:
```
sudo docker exec -it api-service /bin/bash
```



#### Configure Nginx file for Web Server
* Create nginx.conf file for defaults routes in web server

#### Setup Webserver on the Compute Instance
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```
Once the command runs go to `http://<External IP>/` 

### 2.3. (optional) **Delete the Compute Instance / Persistent disk**
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=absent
```
---


### 2.4. Debugging Containers

To debug any of the containers:

* View running containers
```
sudo docker container ls
```

* View images
```
sudo docker image ls
```

* View logs
```
sudo docker container logs api-service -f
sudo docker container logs frontend -f
sudo docker container logs nginx -f
```

* Get into shell
```
sudo docker exec -it api-service /bin/bash
sudo docker exec -it frontend /bin/bash
sudo docker exec -it nginx /bin/bash
```
<a id="kubernetes"></a>
## 3. Deployment with Scaling using Kubernetes
[Return to Table of Contents](#contents)

In the following section we list the steps to deploy DermAID to a K8s cluster on GCP.

<a id="step1"></a>

### 3.1 Step 1 - enable GCP APIs
We need to enable all the APIs in GCP that are needed for K8. For that, search for each of these in the GCP search bar and click enable to enable these API's

* Compute Engine API
* Service Usage API
* Cloud Resource Manager API
* Google Container Registry API
* Kubernetes Engine API

<a id="step2"></a>

### 3.2 Step 2 - Start Deployment Docker Container
We need to start the docker container with the image for creating the K8 clusters. For that:

-  change directory to deployment by

    ```bash
    cd deployment
    ```
- start docker container   
    unix:
    ```bash
    sh docker-shell.sh
    ```
    
    windows   

    ```batch
    docker-shell.bat
    ```

- Check versions of tools
    ```bash 
    gcloud --version
    kubectl version
    kubectl version --client
    ```

- Check if make sure you are authenticated to GCP
    ```bash
    gcloud auth list
    ```

<a id="step3"></a>

### 3.3 Step 3 - Create & Deploy Cluster

```
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=present
```
#### View the App
* Copy the `nginx_ingress_ip` from the terminal from the create cluster command
* Go to `http://<YOUR INGRESS IP>.sslip.io`

        **Note**: Using sslip.io in conjunction with your ingress IP address in Kubernetes on GCP (Google Cloud Platform) is a method to simplify the process of mapping a public IP address to a domain name, particularly when you don't own a domain or don't want to configure DNS settings for a small project or a temporary environment.

#### Delete Cluster
```bash
ansible-playbook deploy-k8s-cluster.yml -i inventory.yml --extra-vars cluster_state=absent
```



<a id="dependencies"></a>
## 4. File Structure and dependencies
[Return to Table of Contents](#contents)


- (01)  Container **api-service.** This is the backend of the application. It comprises  a basic API wrap using FastAPI ato expose the CloudRun for Model Prediction and Saliency Maps. This wrapper follows the recommedations in class to no expose GCP endpoints directly (Vertex/CloudRun/CludFunction, etc). It composed by:

	* *model.py* contains the logic to format the input data and makes the call to the remote API
	* *service.py*, provide the different endpoints for the API
	* *simple_call.py*, a demo script on how to call the API

- (02) Container **frontend-react.** This is a basci web application developed using JavaScript with react as framework. It calls the local api in container **api-services** using the reverse proxy in the local machine, configured to redirect the calls.

- (03) Container **deployment.** Allows an automated deployment using Ansible Playbooks, orchestrated from within a docker container. The playbooks are:

	* deployment/deploy-create-instance.yml
	* deployment/deploy-docker-images.yml
	* deployment/deploy-provision-instance.yml
	* deployment/deploy-setup-containers.yml
	* deployment/deploy-setup-webserver.yml
	* deployment/inventory.yml

- (04) Webserver Nginx and Config file nginx-conf, that enables to redirect the calls to the correspondent docker container, acting as a reverse proxy.  F


<a id="prerequisites"></a>
## 5. Prerequisites
[Return to Table of Contents](#contents)  

Before deploying, you can test locally that the services are running correctly:

* api-service: `http://localhost:9000/docs` make sure you can see the API Docs
* frontend-react: `http://localhost:3000/`  make sure you can see the prediction page
* Docker
* Google Cloud Storage Python SDK
* Access to Google Cloud Services with a secret JSON key file.
* Cloud Run deployment
* Following containers are working correct locally

	api-service


<a id="challenges"></a>
## 6. Challenges and Solutions

### Challenge: Resource Quota Limitations on Cloud Run

**Context:** Once our endpoint was transitioned to Cloud Run, we faced the significant challenge of ensuring that the resources consumed by the service did not exceed our allocated quota.

**Solution:** To address this, we are frequently monitoring the server's performance, especially focusing on response times. This allowed us to anticipate potential issues, adjust as needed, and ensure consistent service availability. As of this update, we have successfully maintained uninterrupted service, ensuring our users can rely on our platform's efficiency and reliability.

So far, the service had been available without issues.

[Return to Table of Contents](#contents)
