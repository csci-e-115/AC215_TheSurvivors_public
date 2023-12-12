import time
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping

# TF Optimization
import tensorflow_model_optimization as tfmot

import wandb
from wandb.keras import WandbCallback


# Random seed
rs = 123

def configure_gpus(num_gpus: int):
    """Function to set number of GPUs for model training

    Args:
        num_gpus (int): Selected number of GPUs by user

    Returns:
        tf.distribute.strategy: strategy for using GPUs
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus or num_gpus == 0:
        # If no GPU available or user selects not to use GPUs -- Use CPU
        print(f"Running on CPU - Avialable number of GPUS: {len(gpus)}")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        strategy = tf.distribute.get_strategy()
        num_used_gpus = 0
    elif gpus == 1 or num_gpus == 1:
        print(f"Running on 1 GPU - Avialable number of GPUS: {len(gpus)}")
        strategy = tf.distribute.get_strategy()
        num_used_gpus = 1
    elif len(gpus) < num_gpus:
        # If number of selected GPUs by user is more than available GPUs -- Use max available GPUs
        print(f"Running on {len(gpus)} GPUs - Avialable number of GPUS: {len(gpus)}")
        device_names = ["/device:GPU:" + gpu.name.split(":")[-1] for gpu in gpus]
        strategy = tf.distribute.MirroredStrategy(devices=device_names)
        num_used_gpus = len(gpus)
    else:
        # If number of selected GPUs by user is less than or equal to available GPUs -- Use what user selected
        print(f"Running on {num_gpus} GPUs - Avialable number of GPUS: {len(gpus)}")
        tf.config.set_visible_devices(gpus[0:num_gpus], "GPU")

        device_names = ["/device:GPU:" + gpu.name.split(":")[-1] for gpu in gpus[0:num_gpus]]
        strategy = tf.distribute.MirroredStrategy(devices=device_names)
        num_used_gpus = num_gpus

    return strategy, num_used_gpus

def parse_tfrecord_example(example_proto):
    # Read TF Records
    num_classes = 8
    num_metadata = 28
    num_channels = 3
    image_height = 256
    image_width = 256
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'metadata': tf.io.FixedLenFeature([num_metadata], tf.int64),
        'label': tf.io.FixedLenFeature([num_classes], tf.int64)
    }

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Image
    image = tf.io.decode_raw(parsed_example['image'], tf.uint8)
    image.set_shape([num_channels * image_height * image_width])
    image = tf.reshape(image, [image_height, image_width, num_channels])
    # Metadata
    metadata = tf.cast(parsed_example['metadata'], tf.int64)
    # Label
    label = tf.cast(parsed_example['label'], tf.int64)

    return (image, metadata), label


def load_tfrecords(tfrecords_dir: str, batch_size: int, paths: dict):
    """Function to load, parse, scale TFRecords and apply augmentation on training data

    Args:
        tfrecords_dir (str): path to tfrecord location on GCP ()
        batch_size (int): batch size for loading data
        paths (dict): dict containing information of which versioned files to use for training

    Returns:
        tf.dataset: tf datasets for training
        tf.dataset: tf datasets for validation
        tf.dataset: tf datasets for testing
    """

    # Get list of all TFRecords

    train_data_list = paths['train']
    val_data_list = paths['val']
    test_data_list = paths['test']

    # Normalization layer
    rescaling = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1 / 255)
    ])

    # Augmentation layer
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=rs),
        tf.keras.layers.RandomRotation(0.5, fill_mode="nearest", seed=rs),
        tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=rs),
        tf.keras.layers.RandomZoom(0.1, fill_mode="nearest", seed=rs)
    ])

    train_ds = tf.data.TFRecordDataset(train_data_list)
    train_ds = train_ds.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: ((rescaling(x[0], training=True), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: ((data_augmentation(x[0], training=True), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = tf.data.TFRecordDataset(val_data_list)
    val_ds = val_ds.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: ((rescaling(x[0], training=True), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.data.TFRecordDataset(test_data_list)
    test_ds = test_ds.map(parse_tfrecord_example, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: ((rescaling(x[0], training=True), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


class CustomLayer(tf.keras.layers.Layer):
    """
    Custom layer inherting from tf layer class, this is used to include additional features in the model
    """

    def build(self, input_shape):
        self.scale = tf.Variable([0.5, 0.5], trainable=True)

    def call(self, inputs):
        scale_norm = tf.nn.softmax(self.scale)
        return tf.tensordot(scale_norm, inputs, axes=1)



def build_model(pretrained_model, metadata_encoded_size, fcnn_nodes, model_name):
    """Function to build the model skeleton

    Args:
        metadata_encoded_size (_type_): dimension of metadata space
        fcnn_nodes (_type_): number of nodes for fcnn model used for metadata
        model_name (_type_): model name given by user

    Returns:
        tf.model: tensorflow model
    """

    # Input to the model
    input_metadata = tf.keras.Input(shape=(metadata_encoded_size))

    # model = DenseNet121(include_top=False, input_shape=(256, 256, 3))
    model = pretrained_model

    # Hidden layers
    hidden = model.output
    hidden = GlobalAveragePooling2D()(hidden)

    # Two different outputs (from image input and age/gender data)
    output = Dense(units=8, activation="softmax")(hidden)
    hidden2 = Dense(units=fcnn_nodes, activation="relu")(input_metadata)
    output2 = Dense(units=8, activation="softmax")(hidden2)

    # Custom layer to stack two outputs
    output = CustomLayer()(tf.stack([output, output2]))
    model = Model(inputs=(model.inputs, input_metadata), outputs=output,  name=model_name)

    return model


def get_class_weights(labels_fp, bucket):
    # Get labels from GCS
    blob = bucket.blob(labels_fp)
    labels = blob.download_as_string()
    labels = np.frombuffer(labels).reshape((-1,8))
    # Convert labels from one-hot encoded to integers
    labels = np.argmax(labels, axis=1)
    # Calculate class weights
    class_labels = np.unique(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=labels)
    return dict(zip(class_labels, class_weights))


def train_model(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    fcnn_nodes: int,
    experiment_tracking: bool,
    labels_fp,
    bucket,
    loss,
    prune: bool,
    strategy
):
    """Function that trains model and communicate with WANDB for experiment tracking

    Args:
        model (Model): model to use for training
        train_ds (tf.data.Dataset): training dataset including images, metadata, and labels
        val_ds (tf.data.Dataset): validation dataset including images, metadata, and labels
        batch_size (int): batch size
        learning_rate (float): learning rate
        epochs (int): number of epochs
        fcnn_nodes (int): number of fcnn nodes for metadata model
        experiment_tracking (bool): whether use experiment tracking during training
        strategy (_type_): strategy for GPU use

    Returns:
        tf.model: trained model for saving
    """

    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    # Early stopping
    es = EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )
    if prune:
        callbacks = [es, tfmot.sparsity.keras.UpdatePruningStep()]
    else:
        callbacks = [es]

    if experiment_tracking:
        wandb.init(
        project = 'dermaid',
        config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model_name": model.name,
        "fcnn_nodes": fcnn_nodes
        },
        name = model.name
        )
        callbacks.append(WandbCallback())

    # Compile the model
    with strategy.scope():
        model.compile(
            loss=loss,
            optimizer=opt,
            weighted_metrics=["categorical_accuracy"],  # Using weighted metrics to reflect class imbalance
        )

        # Train model
        start_time = time.time()

        # Get class weights
        weights = get_class_weights(labels_fp, bucket)
        print("Applying class weights:")
        print(weights)

        # Fit the model and store details in history
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            class_weight=weights,
        )

    execution_time = (time.time() - start_time)/60.0
    print("Training execution time (mins)",execution_time)

    if experiment_tracking:
        # Update W&B
        wandb.config.update({"execution_time": execution_time})
        # Close the W&B run
        wandb.run.finish()

    return model