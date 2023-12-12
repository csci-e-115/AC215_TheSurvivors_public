from google.cloud import storage
from tqdm import tqdm
from zipfile import ZipFile
import argparse
import os
import requests
import shutil

# read env variables
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

# Set file paths
raw_imgs_fp = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
metadata_fp = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
ground_truth_fp = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

# Define sample image set to be used during app development
sample_start = 26000
sample_end = 26500

# Set GCS variables
bucket_name = GCS_BUCKET_NAME

# Get Project Name from enviroment variable
storage_client = storage.Client()

# Get reference to bucket
bucket = storage_client.bucket(bucket_name)

get_file_name = lambda path: os.path.split(path)[1]

# Generate list of sample images
def create_sample_list(start, end):
    sample_list = []
    for i in range(start, end):
        prefix = 'ISIC_2019_Training_Input/ISIC_' + '0' * (7 - len(str(i)))
        image_name = f"{prefix}{i}.jpg"
        sample_list.append(image_name)
    return sample_list

def download_file(source_fp, dest_folder):
    fname = get_file_name(source_fp)
    if not os.path.exists(dest_folder):
        # Create a new directory because it does not exist
        os.makedirs(dest_folder)
        print("New directory created")
    if not os.path.exists(f"{dest_folder}/{fname}"):
        print(f"Downloading {fname} to {dest_folder}")
        r = requests.get(source_fp, verify=True, stream=True)
        r.raw.decode_content = True
        with open(f"{dest_folder}/{fname}", 'wb') as f:
            shutil.copyfileobj(r.raw, f)


def upload_file(source_fp, dest_folder):
    # Destination path in GCS
    fname = get_file_name(source_fp)
    destination_blob_name = f"{dest_folder}/{fname}"
    dst_blob = bucket.blob(destination_blob_name)
    if not dst_blob.exists():
        print(f"Uploading {fname} to {dest_folder}")
        dst_blob.upload_from_filename(source_fp)


def unzip(zip_fp, dest_folder, sample_list=None):
    with ZipFile(zip_fp, 'r') as myzip:
        # Check which files still need to be uploaded. This is checked for all files instead
        # of individually for each file to reduce the number of server requests and save time
        zip_file_list = myzip.namelist()
        gcs_blob_list = bucket.list_blobs(prefix=f"{dest_folder}/")
        gcs_file_list = [get_file_name(blob.name) for blob in gcs_blob_list]
        if not sample_list:
            upload_list = [fp for fp in zip_file_list
                           if get_file_name(fp) not in gcs_file_list and '.jpg' in fp.casefold()]
        else:
            upload_list = [fp for fp in sample_list
                           if fp in zip_file_list and get_file_name(fp) not in gcs_file_list]
        # Upload remaining files
        print(f"Unzipping {zip_fp} to {dest_folder} on GCS")
        for file in tqdm(upload_list):
            fname = get_file_name(file)
            dst_blob = bucket.blob(f'{dest_folder}/{fname}')
            content = myzip.read(file)
            tmp_file = open("tmp_file", 'wb')
            tmp_file.write(content)
            tmp_file.close()
            with open("tmp_file", "rb") as my_file:
                dst_blob.upload_from_file(my_file)
        # Clean up
        if os.path.exists("tmp_file"):
            os.remove("tmp_file")



def main(args=None):
    # Download relevant ISIC files to harddisk
    for fp in [raw_imgs_fp, metadata_fp, ground_truth_fp]:
        fname = get_file_name(fp)
        download_file(source_fp=fp, dest_folder="isic_data")
        if fname[-3:] == "zip":
            # Unzip image files directly to GCS
            if args.sample:
                # Generate sample list and unzip only sample images
                sample_list = create_sample_list(sample_start, sample_end)
                unzip(zip_fp="isic_data/ISIC_2019_Training_Input.zip", dest_folder="raw_images", sample_list=sample_list)
            else:
                # Unzip entire dataset
                unzip(zip_fp="isic_data/ISIC_2019_Training_Input.zip", dest_folder="raw_images")
        # Upload ISIC files to Google Cloud Service
        upload_file(source_fp=f"isic_data/{fname}", dest_folder="isic_data")



if __name__ == '__main__':
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description='Download ISIC data, extract images and upload to GCP')

    parser.add_argument("-s", "--sample", action='store_true',
                        help="Download sample image files rather than full dataset")

    args = parser.parse_args()

    main(args)
