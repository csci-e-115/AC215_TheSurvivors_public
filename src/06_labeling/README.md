# Dermaid Data Labeling Project

<a id="contents"></a>
## Table of Contents

1. [Description](#deployment)
2. [Usage](#usage)
3. [Installation](#installation)
4. [File Structure and dependencies](#dependencies)
5. [Prerequisites](#prerequisites)
6. [Challenges and Solutions](#challenges)

<a id="description"></a>
## 1. Description
[Return to Table of Contents](#contents)

This container handles our app's second front-end for our other users i.e. Dermatologists, using Label Studio. Dermatologists will be able to use Label Studio to assign the correct labels to different images uploaded by the actual end users of our web app.
Iput to the container are unlabeled patient images and environment variables for Google Cloud Services. 
Output from the container are labeled patient images stored in Google Cloud Storage.


This project consists of two Docker containers:

1. `06-dermaid-data-label-cli`: A custom CLI application for managing data labeling tasks.
2. `06-dermaid-data-label-studio`: A Label Studio instance for the actual labeling of patient images.

The CLI application uses pipenv for dependency management. Label Studio uses Google Cloud Services to store labeled and unlabeled patient images.

<a id="description"></a>
## 2. Installation
[Return to Table of Contents](#contents)

### Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### Create Docker Network

If you haven't created the Docker network yet, you can do so by running:

```bash
docker network create dermaid-data-labeling-network
```

### Build and Run Containers

Run the shell script to build the Docker image for the CLI application and start all containers:

```bash
sh ./docker-shell.sh
```

<a id="usage"></a>
## 2. Usage
[Return to Table of Contents](#contents)

### CLI Application

After the containers are up and running, you can interact with the CLI application. More details on how to use the CLI application will be added here.

### Label Studio

Open your web browser and navigate to `http://localhost:8282` to access Label Studio.

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to the Google Cloud credentials file.
- `GCP_PROJECT`: Google Cloud Project ID.
- `GCP_ZONE`: Google Cloud Zone.
- `GCS_BUCKET_NAME`: Google Cloud Storage bucket name.
- `LABEL_STUDIO_URL`: URL for the Label Studio instance.
- `LABEL_STUDIO_USERNAME`: Username for Label Studio.
- `LABEL_STUDIO_PASSWORD`: Password for Label Studio.

<a id="dependencies"></a>
## 4. File structure and dependencies
[Return to Table of Contents](#contents)

(01) `src/06_labeling/label_studio.py` -  Here we have the CLI application that manages the data labeling tasks. It interacts with Label Studio and Google Cloud Services to facilitate the labeling process.

(02) `src/06_labeling/Pipfile` - We used the following packages to support the CLI application:
- `google-cloud-storage` - for interacting with Google Cloud Storage
- `label-studio-sdk` - to run Label Studio

(03) `src/06_labeling/Pipfile.lock` - See description of Pipfile under (02).

(04) `src/06_labeling/Dockerfile` - The dockerfile uses `python:3.8-slim-buster`. It installs dependencies in the Docker container and uses secrets to connect to GCS.

- To build Dockerfile: `docker build -t labeling -f Dockerfile .`
- To run Dockerfile:  `docker run --rm -ti --mount type=bind,source="$(pwd)",target=/app labeling`

(05) `docker-shell.sh` or `docker-shell.bat`- This shell script automates the process of setting up the Docker environment for the `06-dermaid-data-label-cli` container. It performs the following tasks:
- Checks for the existence of the Docker network `dermaid-data-labeling-network` and creates it if necessary.
- Builds the Docker image for the CLI application using the specified Dockerfile.
- Runs the Docker Compose services, specifically targeting the `06-dermaid-data-label-cli` service.
- `docker-shell.sh` is for unix systems and `docker-shell.bat` is for Windows

(06) `docker-compose.yml` - specify how the two Docker environments, `06-dermaid-data-label-cli` and `06-dermaid-data-label-studio` need to be created. The first one depends on the creation of the second.


<a id="prerequisites"></a>
## 5. Prerequisites
[Return to Table of Contents](#contents)

- Docker
- Docker Compose
- Google Cloud Services account and credentials


<a id="challenges"></a>
## 6. Challenges and Solutions
[Return to Table of Contents](#contents)

### Challenge 1: Capturing Additional Metadata

**Problem**: 
Dermatologists, beyond just labeling the images, often require the ability to input related metadata like the patient's age, sex, and other details. This additional layer of information could be invaluable for ensuring the completeness and accuracy of our labeled dataset.

**Solution**: 
We're currently researching the capability of Label Studio to incorporate such metadata input. If Label Studio doesn't natively support this, we might consider:
  - Customizing Label Studio by tapping into its extensible features.
  - Exploring alternative platforms with such capabilities.
  - Building a complementary web application that allows dermatologists to add metadata after labeling, ensuring it's user-friendly and seamlessly integrates with our existing workflow.

### Challenge 2: Hosting Strategy for Label Studio

**Problem**: 
Determining the optimal hosting solution for Label Studio is essential for scalability, security, and overall performance. The choice between options like Cloud Run, virtual machines, and others can impact the overall user experience for our dermatologists and the cost-effectiveness for our operations.

**Solution**: 
To make an informed decision:
  - We are comparing the pros and cons of different hosting solutions, considering factors like cost, ease of deployment, scalability, and security.
  - A pilot test where we deploy Label Studio on multiple platforms and measure performance and cost would be ideal, however, this would be out of scope of the current course.

### Challenge 3: Adapting or Replacing the Labeling Tool

**Problem**: 
If Label Studio cannot be adapted to our specific needs, especially in terms of capturing additional metadata, we might have to look for alternatives or build our solution.

**Solution**: 
If we decide to build our web application (maybe out of scope for CSC115):
  - We would draft a detailed specification, keeping in mind the requirements and preferences of the dermatologists.
  - Consider utilizing existing frameworks or tools as a base to expedite the development process.
  - Ensure the new application is designed to integrate smoothly with our existing ecosystem, especially with Google Cloud Services.
