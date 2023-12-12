import time

import numpy as np


import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.applications import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras import backend as K
from keras.callbacks import EarlyStopping
# TF Optimization
import tensorflow_model_optimization as tfmot


import wandb
from wandb.keras import WandbCallback

from utilities.utils import CustomLayer, get_class_weights


def prune_model(model, train_ds, batch_size, epochs):
    # Define model for pruning
    num_records = train_ds.cardinality()
    end_step = np.ceil(num_records / batch_size).astype(np.int32) * epochs

    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                   final_sparsity=0.80,
                                                                   begin_step=0,
                                                                   end_step=end_step)
    }
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    return model_for_pruning

class Distiller(Model):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        Lambda=0.1,
        temperature=3,
    ):
        """
        optimizer: Keras optimizer for the student weights
        metrics: Keras metrics for evaluation
        student_loss_fn: Loss function of difference between student predictions and ground-truth
        distillation_loss_fn: Loss function of difference between soft student predictions and soft teacher predictions
        lambda: weight to student_loss_fn and 1-alpha to distillation_loss_fn
        temperature: Temperature for softening probability distributions. Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

        # hyper-parameters
        self.Lambda = Lambda
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher (professor)
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.Lambda * student_loss + (1 - self.Lambda) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def build_teacher_model(metadata_encoded_size, fcnn_nodes, model_name="teacher_model"):
    """Function to build the teacher model skeleton

    Args:
        metadata_encoded_size (int): dimension of metadata space
        fcnn_nodes (int): number of nodes for fcnn model used for metadata
        model_name (str): model name given by user

    Returns:
        tf.model: tensorflow model
    """

    # Input to the model
    input_metadata = tf.keras.Input(shape=(metadata_encoded_size))
    model = DenseNet121(
        include_top=False, input_shape=(256, 256, 3), weights="imagenet"
    )

    for layer in model.layers:
        layer.trainable = False

    # Hidden layers
    hidden = model.output
    hidden = Dense(
        units=64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l1(0.02),
        bias_regularizer=keras.regularizers.l1(0.02),
    )(hidden)
    hidden = GlobalAveragePooling2D()(hidden)

    # Two different outputs (from image input and age/gender data)
    output = Dense(units=8, activation="softmax")(hidden)
    hidden2 = Dense(units=fcnn_nodes, activation="relu")(input_metadata)
    output2 = Dense(units=8, activation="softmax")(hidden2)

    # Custom layer to stack two outputs
    output = CustomLayer()(tf.stack([output, output2]))
    model = Model(
        inputs=(model.inputs, input_metadata), outputs=output, name=model_name
    )

    return model


def build_student_model(metadata_encoded_size, fcnn_nodes, model_name="student"):
    """Function to build the student model skeleton

    Args:
        metadata_encoded_size (int): dimension of metadata space
        fcnn_nodes (int): number of nodes for fcnn model used for metadata
        model_name (str): model name given by user

    Returns:
        tf.model: tensorflow model
    """
    # Model input
    input_metadata = tf.keras.Input(shape=(metadata_encoded_size))

    cnn_model = Sequential(
        [
            keras.Input(shape=[256, 256, 3]),
            keras.layers.Conv2D(
                filters=8,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=keras.initializers.GlorotUniform(seed=1212),
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            keras.layers.Conv2D(
                filters=16,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer=keras.initializers.GlorotUniform(seed=2121),
            ),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),
            keras.layers.Flatten(),
        ],
        name=model_name,
    )

    # Hidden layers
    hidden = cnn_model.output
    hidden = GlobalAveragePooling2D()(hidden)

    # Two different outputs (from image input and age/gender data)
    output = Dense(units=8, activation="softmax")(hidden)
    hidden2 = Dense(units=fcnn_nodes, activation="relu")(input_metadata)
    output2 = Dense(units=8, activation="softmax")(hidden2)

    # Custom layer to stack two outputs
    output = CustomLayer()(tf.stack([output, output2]))
    student_model = Model(
        inputs=(cnn_model.inputs, input_metadata), outputs=output, name=model_name
    )

    return student_model


def distill_teacher_to_student(
    teacher_model: Model,
    student_model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    fcnn_nodes: int,
    experiment_tracking: bool,
    labels_fp,
    bucket,
    student_loss,
    distillation_loss,
    strategy
):
    """distillation function

    Args:
        teacher_model (Model): teacher model
        student_model (Model): student model
        train_ds (tf.data.Dataset): training dataset
        val_ds (tf.data.Dataset): validation dataset
        batch_size (int): batch size
        learning_rate (float): learning rate
        epochs (int): number of epochs
        fcnn_nodes (int): number of FCNN nodes for metadata
        experiment_tracking (bool): If experiment tracking should be used
        labels_fp (_type_): labels
        bucket (GCP bucket): bucket name
        student_loss (float): loss of student model
        distillation_loss (float): loss of distillation model
        strategy (_type_): TF strategy

    Returns:
        Model: distillation model
    """
    Lambda = 0.75
    temperature = 12

    # Free up memory
    K.clear_session()

    # Build the distiller model
    distiller_model = Distiller(teacher=teacher_model, student=student_model)

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Early stopping
    es = EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )
    callbacks = [es]

    if experiment_tracking:
        wandb.init(
            project="dermaid",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "model_name": student_model.name,
                "fcnn_nodes": fcnn_nodes,
            },
            name=student_model.name,
        )
        callbacks.append(WandbCallback())

    with strategy.scope():
        # Compile
        distiller_model.compile(
            optimizer=optimizer,
            student_loss_fn=student_loss,
            distillation_loss_fn=distillation_loss,
            metrics=[keras.metrics.CategoricalAccuracy()],
            Lambda=Lambda,
            temperature=temperature,
        )

        # Distill teacher to student
        start_time = time.time()

        # Get class weights
        weights = get_class_weights(labels_fp, bucket)
        print("Applying class weights:")
        print(weights)

        training_results = distiller_model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1,
            class_weight=weights,
        )
    execution_time = (time.time() - start_time) / 60.0
    print("Training execution time (mins)", execution_time)

    if experiment_tracking:
        # Update W&B
        wandb.config.update({"execution_time": execution_time})
        # Close the W&B run
        wandb.run.finish()

    return distiller_model



def custom_distributed_training(teacher_model, student_model, train_ds, val_ds, strategy, epochs, learning_rate, experiment_tracking, batch_size, fcnn_nodes):
    """Distillation model for MirroredStrategy (more than 1 GPU)

    Args:
        teacher_model (Model): teacher model
        student_model (Model): student model
        train_ds (tf.data.Dataset): training dataset
        val_ds (tf.data.Dataset): validation dataset
        strategy (_type_): TF strategy
        epochs (int): number of epochs
        learning_rate (float): learning rat
        experiment_tracking (bool): If experiment tracking should be used
        batch_size (int): batch size
        fcnn_nodes (int): number of FCNN nodes for metadata


    Returns:
        Model: distillation model
    """
    Lambda = 0.75
    temperature = 12
    opt = keras.optimizers.Adam(learning_rate=learning_rate)

    
    with strategy.scope():
        # Set reduction to `NONE` so you can do the reduction yourself.
        student_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)
        distillation_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE)
        def compute_student_loss(labels, predictions, model_losses):
            per_example_loss = student_loss_fn(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss
        def compute_teacher_loss(labels, predictions, model_losses):
            per_example_loss = distillation_loss_fn(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss)
            if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
            return loss


    with strategy.scope():
        val_loss = tf.keras.metrics.Mean(name='val_loss')

        train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='categorical_accuracy')
        val_accuracy = tf.keras.metrics.CategoricalAccuracy(
            name='val_categorical_accuracy')

    def train_step(data):

        teacher_predictions = teacher_model(data[0], training=False)

        with tf.GradientTape() as tape:
            student_predictions = student_model(data[0], training=True)

            student_loss = compute_student_loss(data[1], student_predictions, student_model.losses)

            distillation_loss = compute_teacher_loss(
                tf.nn.softmax(teacher_predictions / temperature, axis=1),
                tf.nn.softmax(student_predictions / temperature, axis=1),
                teacher_model.losses,
            )
            loss = Lambda * student_loss + (1 - Lambda) * distillation_loss

        gradients = tape.gradient(loss, student_model.trainable_variables)
        opt.apply_gradients(zip(gradients, student_model.trainable_variables))

        train_accuracy.update_state(data[1], student_predictions)
        return loss

    def test_step(data):

        predictions = student_model(data[0], training=False)
        t_loss = compute_student_loss(data[1], predictions, student_model.losses)

        val_loss.update_state(t_loss)
        val_accuracy.update_state(data[1], predictions)


    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs,))  


    if experiment_tracking:
        wandb.init(
            project="dermaid",
            config={
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "model_name": student_model.name,
                "fcnn_nodes": fcnn_nodes,
            },
            name=student_model.name,
        )
   
    start_time = time.time()

    for epoch in range(epochs):
        # TRAIN LOOP
        total_loss = 0.0
        if epoch == 0:
            max_batch = "unknown"
        else:
            max_batch = num_batches
        num_batches = 0
        for x in train_ds:
            total_loss += distributed_train_step(x)
            num_batches += 1
            print(f"{num_batches}/{max_batch} -- loss: {total_loss/num_batches}")
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in val_ds:
            distributed_test_step(x)    


        template = ("Epoch {}, loss: {}, categorical_ccuracy: {}, val_Loss: {}, "
                    "val_categorical_accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                                train_accuracy.result(), val_loss.result(),
                                val_accuracy.result()))

        val_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

    execution_time = (time.time() - start_time) / 60.0
    print("Training execution time (mins)", execution_time)

    if experiment_tracking:
        # Update W&B
        wandb.config.update({"execution_time": execution_time})
        # Close the W&B run
        wandb.run.finish()

    return student_model

