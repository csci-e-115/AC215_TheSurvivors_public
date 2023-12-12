# Data Preprocessing

This container is the second part of our data pipeline. It collects the raw data from GCS, converts it into tf Datasets and then stores the complete datasets as tfrecords, back into GCS.

The tfrecords created in this container are also versioned.

<a id="contents"></a>
## Table of Contents

1. [Deployment](#1-deployment)
2. [File structure and dependencies](#2-file-structure-and-dependencies)
3. [Challenges & Solutions](#3-challenges--solutions)
4. [Versioning Showcase](#4-versioning-showcase)

## 1. Deployment

To deploy this container run

```bash
  sh docker-shell.sh`
```
Commands to run inside docker container to execute processing of data and converting into tf records:

```
# Preprocessing for all raw images from ISIC website
python preprocess_data.py -r

# Preprocessing for labelled images from dermatologists
python preprocess_data.py -d
```

The container has the following requirements:

1. `git.env` file: An environment file with the following environment variables:
    ```python
    GIT_URL=https://<username>:<password>@github.com/csci-e-115/AC215_TheSurvivors.git
    GIT_BRANCH=<name of the branch to use for versionining commits>
    GIT_USERNAME=<username>
    GIT_USEREMAIL=<email of the user>
    GIT_PASSWORD=<password>
    ```
1. `secrets` folder with the json credentials for GCP -- File name expected is `ac215-dermaid.json`
## Usage

The container performs the following task:
- `preprocess_data.py`: Running a 5-step data pipeline:
   1. Read in the 25,331 dermatological images from GCS OR new images in the labeled images folder
   2. Resize images to a resolution of 256x256
   3. Read in metadata and labels corresponding to images
   4. Create TFRecords, zipping together the images, the metadata and the target labels and store them directly on GCS
   5. Stores the paths to the tfrecords on GCS into a pickle file

      ```bash
      python preprocess_data.py
      ```

- Input to the container are the GCS source and destination folders and GCP secrets (shared via Docker). Also, the desired image resolution and target number of images per shard are required.
- Output from the container are resized training, validation and test sets as TFRecords on GCS. 

## 2. File structure and dependencies

(01) `src/02_preprocess_data/preprocess_data.py` - Here we read the raw images, metadata and labels from GCS into TFRecords. All datasets are uploaded to GCS.

(02) `src/02_preprocess_data/Pipfile` - We used the following packages to support data pre-processing:
- `pandas` - for reading and processing metadata and label files
- `tensorflow` - for generating and processing keras datasets, as well as running data augmentation
- `google-cloud-storage` - for connecting to GCP
- `scikit-learn` - for performing the train-test split of the dataset
- `tqdm` - for displaying the progress of TFRecord generation
- `whylogs` - for model monitoring
- `Pillow` - for image handling

(03) `src/02_preprocess_data/Pipfile.lock` - See description of Pipfile under (02).

(04) `src/02_preprocess_data/Dockerfile` - The dockerfile uses `python:3.8-slim-buster`. It installs dependencies in the Docker container and uses secrets to connect to GCS.

(05) `src/02_preprocess_data/docker-shell.sh` or `src/02_preprocess_data/docker-shell.bat`- This shell script automates the process of setting up the Docker environment.


## 3. Challenges & Solutions

One of the largest challenges in versioning of the datasets was to be able to run versioninig, appropritately, from a container created from a sub-folder of a git repo. Since versioning pushes a commit to the original git repo, we had to load the entire repo into the container to ensure that the git push did not compromise the sanctity of the entire project.

Initially, we tried to create a TF Dataset directly from the raw images stored on GCP. This worked for smaller data samples, but ran out of memory for the entire dataset. By using TF Records, we have a more efficient way to deal with the large dataset and can also combine images, metadata and labels into a single instance.
