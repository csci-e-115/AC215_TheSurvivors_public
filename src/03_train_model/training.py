import argparse

import pickle
from google.cloud import storage

import wandb

from keras.applications import DenseNet121

from utilities.utils import *
from utilities.model_compression import *


# read env variables
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

GCS_DATASETS = "datasets"
GCS_MODELS = "models"

DVC_VERSIONED_FOLDER = "Versioning"
DVC_VERSIONED_FILE = "versioned_paths.pickle"

bucket_name = GCS_BUCKET_NAME
datasets = GCS_DATASETS
models = GCS_MODELS

# Initiate Storage clien3
storage_client = storage.Client()

# Get reference to bucket
bucket = storage_client.bucket(bucket_name)

# Set GCS variables
tfrecords_dir = f"gs://{bucket_name}/tf_records"
model_dir = f"gs://{bucket_name}/{models}"
labels_fp = "isic_data/labels.csv"
teacher_model_dir = f"gs://{bucket_name}/teacher_model"
student_model_dir = f"gs://{bucket_name}/student_model"
distilled_model_dir = f"gs://{bucket_name}/distilled_model"
pruned_model_dir = f"gs://{bucket_name}/pruned_model"

# Distillation parameters
load_teacher = True # Weather to train teacher or load from GCS
student_scratch = False  # Weather to train stand-alone student (just for comparison)
load_distilled = False # Weather to load distilled model from GCS or train it

def main(args):
    # check that versioned data can be downloaded
    # else, throw an error and exit
    assert os.path.exists(
        os.path.join(DVC_VERSIONED_FOLDER, DVC_VERSIONED_FILE)
    ), "No data found to download, Please run `sh dvc-pull.sh` with tag of dataset to fetch"

    # load pickle file with path names to use
    with open(os.path.join(DVC_VERSIONED_FOLDER, DVC_VERSIONED_FILE), "rb") as f:
        paths = pickle.load(f)

    # based on available data
    metadata_encoded_size = 28

    # Get the arguments
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    fcnn_nodes = args.fcnn_nodes
    num_gpus = args.num_gpus
    model_name = args.model_name
    experiment_tracking = args.experiment_tracking
    prune = args.pruning
    distillation = args.distillation


    if experiment_tracking:
        wandb.login()

    strategy, num_used_gpus = configure_gpus(num_gpus)

    with strategy.scope():
        # Load datasets
        train_ds, val_ds, test_ds = load_tfrecords(
            tfrecords_dir=tfrecords_dir, batch_size=batch_size, paths=paths
        )

        pretrained_model = DenseNet121(include_top=False, input_shape=(256, 256, 3))


    if not distillation:
        loss = keras.losses.CategoricalCrossentropy()
        with strategy.scope():
            if prune:
                pretrained_model = prune_model(pretrained_model, train_ds, batch_size, epochs)
            model = build_model(pretrained_model, metadata_encoded_size, fcnn_nodes, model_name)
            trained_model = train_model(
                model,
                train_ds,
                val_ds,
                batch_size,
                learning_rate,
                epochs,
                fcnn_nodes,
                experiment_tracking,
                labels_fp,
                bucket,
                loss,
                prune,
                strategy
            )
        # Saving the model to GCS
        print("Uploading the model to GCS...")
        if prune:
            trained_model.save(pruned_model_dir)
        else:
            trained_model.save(model_dir)
    else:
        teacher_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        distillation_loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        student_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

        with strategy.scope():
            if load_teacher:
                trained_teacher_model = tf.keras.models.load_model(teacher_model_dir)
            else:
                teacher_model = build_teacher_model(metadata_encoded_size, fcnn_nodes)
                trained_teacher_model = train_model(
                    teacher_model,
                    train_ds,
                    val_ds,
                    batch_size,
                    learning_rate,
                    10,
                    fcnn_nodes,
                    experiment_tracking,
                    labels_fp,
                    bucket,
                    teacher_loss,
                    prune,
                    strategy
                )

                # Saving the model to GCS
                print("Uploading the teacher model to GCS...")
                trained_teacher_model.save(teacher_model_dir)

            if student_scratch:
                student_model = build_student_model(metadata_encoded_size, fcnn_nodes)
                trained_student_model = train_model(
                    student_model,
                    train_ds,
                    val_ds,
                    batch_size,
                    learning_rate,
                    10,
                    fcnn_nodes,
                    experiment_tracking,
                    labels_fp,
                    bucket,
                    student_loss,
                    prune,
                    strategy
                )

            if load_distilled:
                distiller_model = tf.keras.models.load_model(distilled_model_dir)
            else:

                student_distill_model = build_student_model(
                    metadata_encoded_size, fcnn_nodes, model_name="student_distill"
                )
                if num_used_gpus < 2:
                    distiller_model = distill_teacher_to_student(
                        trained_teacher_model,
                        student_distill_model,
                        train_ds,
                        val_ds,
                        batch_size,
                        learning_rate,
                        10,
                        fcnn_nodes,
                        experiment_tracking,
                        labels_fp,
                        bucket,
                        student_loss,
                        distillation_loss,
                        strategy,
                    )
                else:
                    if experiment_tracking:
                        print("NOTE: WANDB will only track system resorces when running distillation with 2+ GPUs")
                    distiller_model = custom_distributed_training(
                        trained_teacher_model,
                        student_distill_model,
                        train_ds,
                        val_ds,
                        strategy,
                        epochs,
                        learning_rate,
                        experiment_tracking,
                        batch_size,
                        fcnn_nodes,
                    )

                # Saving the model to GCS
                print("Uploading the distilled model to GCS...")
                distiller_model.save(distilled_model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Input model parameters")
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-f", "--fcnn_nodes", type=int, default=1024)
    parser.add_argument("-n", "--model_name", type=str, default="base_model")
    parser.add_argument("-x", "--experiment_tracking", type=bool, default=True)
    parser.add_argument("-p", "--pruning", type=bool, default=False)
    parser.add_argument("-d", "--distillation", type=bool, default=False)

    args = parser.parse_args()

    main(args)