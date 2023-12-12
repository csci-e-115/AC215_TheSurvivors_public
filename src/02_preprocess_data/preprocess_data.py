from io import StringIO
import pickle
from google.cloud import storage
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse, os, math
import pandas as pd
import tensorflow as tf
import json
import uuid

# whylogs
import whylogs
from whylogs.extras.image_metric import log_image
from whylogs.api.writer.whylabs import WhyLabsWriter
import datetime
from io import BytesIO
from PIL import Image

# read env variables
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
DVC_VERSIONED_FILE = "versioned_paths.pickle"
LABELSTUDIO_TRACKER_FILE = "labelstudio_tracker_file.pickle"

# Set GCS variables
bucket_name = GCS_BUCKET_NAME
labels_fp = "isic_data/ISIC_2019_Training_GroundTruth.csv"
metadata_fp = "isic_data/ISIC_2019_Training_Metadata.csv"
raw_image_dir = f"gs://{bucket_name}/raw_images"
tf_record_dir = f"gs://{bucket_name}/tf_records"
tf_record_dir_labeled = f"gs://{bucket_name}/tf_records_labeled/"
dataset_dir = f"gs://{bucket_name}/datasets"

min_datasize = 10

annotations_dir = f"uploads_annotated"

# Initiate Storage client from environment variable
storage_client = storage.Client()

# Get reference to bucket
bucket = storage_client.bucket(bucket_name)

# Get list of all image file names on GCP
data_list = [blob.name.split("/")[-1] for blob in storage_client.list_blobs(bucket, prefix=f"{raw_image_dir.split('/')[-1]}/ISIC")]

# Set parameters
label_codes = {"MEL": 0, "NV": 1, "BCC": 2, "AK": 3, "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7}
num_classes = len(label_codes)
image_height = image_width = 256
num_channels = 3

# Set random seed
rs = 123

get_file_name = lambda path: os.path.split(path)[1]

def get_features():
    # Filter for images in data directory
    img_list = [x.split(".")[0] for x in data_list]

    # Get labels from GCS
    blob_labels = bucket.get_blob(labels_fp)
    csv_data_labels = blob_labels.download_as_string()
    csv_string_labels = csv_data_labels.decode('utf-8')
    df_labels = pd.read_csv(StringIO(csv_string_labels))
    df_labels = df_labels.drop(columns="UNK")
    df_labels = df_labels.loc[df_labels['image'].isin(img_list)]
    # Upload labels to GCP for downstream use
    blob = bucket.blob('isic_data/labels.csv')
    labels_array = df_labels.drop(columns="image").to_numpy()
    blob.upload_from_string(labels_array.tobytes())

    # Get metadata from GCS and apply one-hot-encoding
    blob_metadata = bucket.get_blob(metadata_fp)
    csv_data_metadata = blob_metadata.download_as_string()
    csv_string_metadata = csv_data_metadata.decode('utf-8')
    df_metadata = pd.read_csv(StringIO(csv_string_metadata)).drop(columns="lesion_id").fillna('')

    df_labels_multiclass = df_labels.copy()
    df_labels_multiclass['ground_truth'] = df_labels.iloc[:, 1:].idxmax(axis=1)
    df_all = df_metadata.merge(df_labels_multiclass[['image', 'ground_truth']], on = 'image', how = 'left')

    df_metadata = pd.get_dummies(
        data=df_metadata.astype('str'),
        columns=["age_approx", "anatom_site_general", "sex"],
        drop_first=True,
        dtype="int",
    )
    df_metadata = df_metadata.loc[df_metadata['image'].isin(img_list)]

    return df_metadata, df_labels, df_all

def prepare_new_data():
    # get list of ids from gcp bucket 

    blob = bucket.blob(LABELSTUDIO_TRACKER_FILE)
    pickle_in = blob.download_as_string()
    completed_list = pickle.loads(pickle_in)

    # completed_list = ['1']
    # get the complete list from the folder on gcp containing the annotations
    annotations_list = [blob.name.split("/")[-1] for blob in storage_client.list_blobs(bucket, prefix=annotations_dir)]

    new_list = [a for a in annotations_list if a not in completed_list]

    # read all these json files
    tasks = {
        'file_id': [],
        'image': [],
        'age_approx': [],
        'anatom_site_general': [],
        'sex': [],
        'ground_truth': [],
    }
    for f in new_list:
        if len(f) == 0: # handle a blank file name that comes for no good reason
            continue

        blob = bucket.get_blob(os.path.join(annotations_dir, f))
        annotation = blob.download_as_string()
        annotation = json.loads(annotation)

        if not annotation['task']['is_labeled']: # dont do anything if this has not been labeled
            continue

        tasks['file_id'].append(f)
        tasks['image'].append(annotation['task']['data']['image_url'])
        tasks['age_approx'].append(annotation['task']['data']['user_age'])
        tasks['anatom_site_general'].append(annotation['task']['data']['user_anatom_site_general'])
        tasks['sex'].append(annotation['task']['data']['user_gender'])
        tasks['ground_truth'].append(annotation['result'][0]['value']['choices'][0])
    
    tasks = pd.DataFrame.from_dict(tasks)

    img_data = tasks['image'].tolist()

    df_metadata = pd.get_dummies(
        data=tasks[["image", "age_approx", "anatom_site_general", "sex"]].astype('str'),
        columns=["age_approx", "anatom_site_general", "sex"],
        drop_first=True,
        dtype="int",
    )

    df_labels = pd.get_dummies(
        data=tasks[["image", "ground_truth"]].astype('str'),
        columns=["ground_truth"],
        drop_first=True,
        dtype="int",
    )

    df_all = tasks[['image', 'age_approx', 'anatom_site_general', 'sex', 'ground_truth']]

    completed_list.extend(tasks['file_id'].tolist())

    # PENDING IMPLEMENTATION: Check that the columns are the same as the original file..
    # Since there might be fewer records in these files, there could be a scenario that some options are missing

    return img_data, df_metadata, df_labels, df_all, completed_list


def create_tf_example(item, df_metadata, df_labels):
    # Get image data
    if 'gs://' != item[:5]:
        image_fp = raw_image_dir.split("/")[-1] + "/" + item
        image_name = item.split(".")[0]
    else:
        image_fp = item.replace(f"gs://{bucket_name}/", "")
        image_name = item # used to get metadata & labels from the dfs

    # Read image
    blob = bucket.get_blob(image_fp)
    image = blob.download_as_bytes()
    # create image profile
    img_profile = log_image(Image.open(BytesIO(image))).profile()

    image = tf.image.decode_jpeg(image, channels=num_channels)
    image = tf.image.resize(image, [image_height, image_width])
    image = tf.cast(image, tf.uint8)

    # Read metadata and label
    metadata = df_metadata[df_metadata.image == image_name].drop(columns="image").values.flatten().tolist()
    label = df_labels[df_labels.image == image_name].drop(columns="image").astype('int64').values.flatten().tolist()

    # Build feature dict
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.numpy().tobytes()])),
        'metadata': tf.train.Feature(int64_list=tf.train.Int64List(value=metadata)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example, img_profile


def create_tf_records(data, shard_size, prefix='', folder='tf_records', df_metadata = None, df_labels = None):
    num_records = len(data)
    num_shards = math.ceil(num_records / shard_size)  # 500 images at reduced size take approx 100MB space

    paths = []
    print(f"Creating {num_shards} shards for {prefix}:")

    # logging with whylogs
    merged_profile = None
    for i in range(0, num_records, shard_size):
        print(f"Starting {i}")
        path = '{}/{}/{:04d}.tfrecords'.format(folder, prefix, i//shard_size)
        paths.append(path)

        # Write the file
        with tf.io.TFRecordWriter(path) as writer:
            # Filter the subset of data to write to tfrecord file
            for item in tqdm(data[i:i+shard_size]):
                tf_example, img_profile = create_tf_example(item, df_metadata, df_labels)
                writer.write(tf_example.SerializeToString())

                print(f"Written to path {path}")

                if merged_profile is None:
                    merged_profile = img_profile.view()
                else:
                    merged_profile = merged_profile.merge(img_profile.view())

            
    print(f"Logging images")
    why_writer = WhyLabsWriter()
    why_writer.write(merged_profile)

    return paths

def main(args=None):
    # Set shard size parameter (default: 500 images per shard)
    shard_size = args.shard_size

    if args.raw_data:
        # get the metadata & labels
        df_metadata, df_labels, df_all = get_features()
        # set data to be used to data_list
        img_data = data_list
        folder = tf_record_dir
    elif args.expert_data is not None:
        img_data, df_metadata, df_labels, df_all, completed_list = prepare_new_data()

        assert len(img_data) > min_datasize, "Oops! Not enough data points for re-training"

        folder = os.path.join(tf_record_dir_labeled, str(uuid.uuid4())) # create a random folder
    else:
        raise NotImplementedError("Missing CLI args")

    date = datetime.datetime.now(datetime.timezone.utc)

    # Log metadata with ground truth
    print("Logging metadata & labels")
    results = whylogs.log(df_all)
    profile = results.profile()
    profile.set_dataset_timestamp(date)
    # print(profile.view().to_pandas())
    why_writer = WhyLabsWriter()
    why_writer.write(file=profile.view())

    # Split data into train, val, test
    val_ratio = 0.15
    test_ratio = 0.15
    print("Splitting data into train, val, test:")
    train_x, valtest_x = train_test_split(img_data, test_size=val_ratio+test_ratio)
    val_x, test_x = train_test_split(valtest_x, test_size=test_ratio/(val_ratio+test_ratio))
    print("train_x count:", len(train_x))
    print("val_x count:", len(val_x))
    print("test_x count:", len(test_x))

    # Create TFRecords and store on GCP
    paths = {}
    paths['train'] = create_tf_records(
        train_x, shard_size=shard_size, prefix='train', folder=folder, df_metadata=df_metadata, df_labels=df_labels
        )
    paths['val'] = create_tf_records(
        val_x, shard_size=shard_size, prefix='val', folder=folder, df_metadata=df_metadata, df_labels=df_labels
        )
    paths['test'] = create_tf_records(
        test_x, shard_size=shard_size, prefix='test', folder=folder, df_metadata=df_metadata, df_labels=df_labels
        )
    
    # if not os.path.exists(DVC_VERSIONED_FOLDER):
    #     os.makedirs(DVC_VERSIONED_FOLDER)

    with open(DVC_VERSIONED_FILE, 'wb') as f:
        pickle.dump(paths, f)

    # upload this to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(DVC_VERSIONED_FILE)
    blob.upload_from_filename(DVC_VERSIONED_FILE)

    # update completed list if required
    if args.expert_data is not None:
        blob = bucket.blob(LABELSTUDIO_TRACKER_FILE)
        pickle_out = pickle.dumps(completed_list)
        blob.upload_from_string(pickle_out)

if __name__ == '__main__':
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(
        description='Create mixed datasets of images, metadata and labels'
    )

    parser.add_argument(
        "-r", "--raw-data",
        action="store_true",
        default=True,
        help="Use raw data from ISIC website"
    )

    parser.add_argument(
        "-e", "--expert-data",
        action="store_true",
        default=False,
        help="Use newly labeled data from experts"
    )

    parser.add_argument(
        "-s", "--shard_size",
        nargs='?',
        type=int,
        default=500,
        help="Set the number of images per shard",
    )

    args = parser.parse_args()

    main(args)
