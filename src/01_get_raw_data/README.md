# Data Download

This container is the first part of our data pipeline. It downloads the zipped images from the ISIC 2019 Challenge website, extracts images and uploads them to GCS. It also downloads metadata and labels and stores them in a GCS bucket.

<a id="contents"></a>
## Table of Contents

1. [Deployment](#1-deployment)
2. [Features](#2-features)
3. [File structure and dependencies](#3-file-structure-and-dependencies)
4. [Challenges & Solutions](#4-challenges--solutions)

<a id="deployment"></a>
## 1. Deployment

To deploy this container run

```bash
  sh docker-shell.sh`
```

The container has the following requirements:

1. `secrets` folder with the json credentials for GCP -- File name expected is `ac215-dermaid.json`

<a id="features"></a>
## 2. Features

The container performs the following tasks:
- This container reads three files from the “ISIC 2019 Challenge” website: The 9.1GB ZIP file of dermatological images, a CSV file with metadata and a CSV file with labels and uploads them to GCS. Images are unpacked and stored in a dedicated folder on GCS.
- Input to the container are the source file URLs, GCS destination folders and GCP secrets (shared via Docker)
- Output from the container are the three original “ISIC 2019 Challenge” files, as well as the 25,331 unpacked JPEGs stored on GCS.

<a id="structure"></a>
## 3. File structure and dependencies

(01) `src/01_get_raw_data/get_data.py` - Here we download the “ISIC 2019 Challenge” files to a local folder before uploading them to GCS. JPGs with dermatological images are individually unpacked and uploaded to a dedicated folder on GCS.

(02) `src/01_get_raw_data/Pipfile` - We used the following packages to support the download, upload and unzipping process:
- `google-cloud-storage` - for interacting with GCP
- `tqdm` - for visualizing progress of unzipping and uploading images to GCP

(03) `src/01_get_raw_data/Pipfile.lock` - See description of Pipfile under (02).

(04) `src/01_get_raw_data/Dockerfile` - The dockerfile uses `python:3.8-slim-buster`. It installs dependencies in the Docker container and uses secrets to connect to GCS.

(05) `src/01_get_raw_data/docker-shell.sh` - To deploy the docker container

<a id="challenges"></a>
## 4. Challenges & Solutions

Since not all files in the zip file hosted on the ISIC 2019 Challenge website are images, the container checks all zipped files individually and extracts only image files. This way, it ensures that only valid images are stored in the raw data folder on GCS.
