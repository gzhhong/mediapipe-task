import tensorflow as tf
from tensorflow import keras

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from config import IMAGE_EDGE  # import constant
import cv2
import math  # add math module
from tensorflow.keras.layers import Rescaling, Layer
from tensorflow.keras.utils import register_keras_serializable

layers = keras.layers
@register_keras_serializable()
class CoordinatesMergeLayer(layers.Layer):
    """custom layer for merging x, y, z coordinates"""
    def call(self, inputs):
        x_coords, y_coords, z_coords = inputs
        # ensure all inputs have the same shape
        x_coords = tf.reshape(x_coords, [-1, 21])
        y_coords = tf.reshape(y_coords, [-1, 21])
        z_coords = tf.reshape(z_coords, [-1, 21])
        # use K.stack instead of tf.stack
        stacked = tf.keras.backend.stack([x_coords, y_coords, z_coords], axis=2)
        return tf.keras.backend.reshape(stacked, [-1, 63])
    
    def get_config(self):
        # add serialization support
        return super().get_config()

@register_keras_serializable()
class NegativeActivation(layers.Layer):
    """custom layer to replace Lambda layer for negative z coordinate conversion"""
    def call(self, inputs):
        return -inputs
    
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
def normalize_image(image):
    """uniform image normalization function"""
    # ensure image is float32 type
    x = tf.cast(image, tf.float32)
    # use (x - mean) / std form for normalization, consistent with metadata
    # mean = [0.0, 0.0, 0.0], std = [255.0, 255.0, 255.0]
    return (x - 0.0) / 255.0

@register_keras_serializable()
class ScaleShiftLayer(Layer):
    def __init__(self, **kwargs):
        super(ScaleShiftLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return -0.5 * (inputs + 1)

    def get_config(self):
        config = super(ScaleShiftLayer, self).get_config()
        return config

def create_mediapipe_model(input_shape=(IMAGE_EDGE, IMAGE_EDGE, 3)):
    """lightweight model architecture"""
    # use lightweight MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=0.5,  # use smaller alpha value to make model lighter
        include_top=False,
        weights='imagenet'
    )
    
    # freeze first 70% layers
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(0.7 * num_layers)]:
        layer.trainable = False
    
    inputs = layers.Input(shape=input_shape, dtype=tf.float32, name='input_image')

    # The key to improve the performance is to scale the input image to [-1, 1]
    x = Rescaling(scale=2.0, offset=-1.0, name="scale_0_1_to_neg1_1")(inputs)
    # extract features through base model
    x = base_model(x)
    
    # feature processing
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # important points branch
    important_branch = layers.Dense(256, activation='relu')(x)
    important_branch = layers.BatchNormalization()(important_branch)
    important_branch = layers.Dense(128, activation='relu')(important_branch)
    important_branch = layers.BatchNormalization()(important_branch)
    
    # main branch
    main_branch = layers.Dense(256, activation='relu')(x)
    main_branch = layers.BatchNormalization()(main_branch)
    main_branch = layers.Dense(128, activation='relu')(main_branch)
    main_branch = layers.BatchNormalization()(main_branch)
    
    # merge branches
    x = layers.Concatenate()([important_branch, main_branch])
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # coordinate prediction branch
    def create_coordinate_branch(name, activation=None):
        coords = layers.Dense(128, activation='relu')(x)
        coords = layers.BatchNormalization()(coords)
        coords = layers.Dense(64, activation='relu')(coords)
        coords = layers.BatchNormalization()(coords)
        coords = layers.Dense(21)(coords)
        if isinstance(activation, str):
            return layers.Activation(activation, name=name)(coords)
        elif activation is not None:
            return activation(name=name)(coords)
        return coords
    
    # predict coordinates
    x_coords = create_coordinate_branch('x_coords', 'sigmoid')
    y_coords = create_coordinate_branch('y_coords', 'sigmoid')

    # Use tanh activation for z_coords to get values in [-1, 1]
    z_coords = create_coordinate_branch('z_coords', 'tanh')

    # Scale and shift z_coords to be in [-1, 0]
    z_coords = ScaleShiftLayer(name='z_activation')(z_coords)

    # merge coordinates
    landmarks = CoordinatesMergeLayer(name='landmarks')([x_coords, y_coords, z_coords])
    
    # auxiliary branch
    presence_score = layers.Dense(64, activation='relu')(x)
    presence_score = layers.BatchNormalization()(presence_score)
    presence_score = layers.Dense(1, activation='sigmoid', name='presence_score')(presence_score)
    
    handedness = layers.Dense(64, activation='relu')(x)
    handedness = layers.BatchNormalization()(handedness)
    handedness = layers.Dense(2, activation='softmax', name='handedness')(handedness)
    
    world_landmarks = layers.Dense(128, activation='relu')(x)
    world_landmarks = layers.BatchNormalization()(world_landmarks)
    world_landmarks = layers.Dense(63, activation='sigmoid', name='world_landmarks')(world_landmarks)
    
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            'landmarks': landmarks,
            'handedness': handedness,
            'presence_score': presence_score,
            'world_landmarks': world_landmarks
        }
    )
    
    return model

def process_image_and_coords(image, coords, target_size=(IMAGE_EDGE, IMAGE_EDGE)):
    """process image and coordinates"""
    
    # resize image to
    image = tf.image.resize(image, target_size)
    
    # use uniform normalization function
    image = normalize_image(image)

    coords = tf.reshape(coords, [-1])
    return image, coords


def parse_tfrecord(example_proto):
    """parse TFRecord data"""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'landmarks': tf.io.FixedLenFeature([63], tf.float32),
        'filename': tf.io.FixedLenFeature([], tf.string)  # add filename feature
    }
    
    # parse example
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # decode image
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    
    # get landmarks
    landmarks = parsed_features['landmarks']
    
    # get filename and determine hand type
    filename = parsed_features['filename']
    is_left_hand = tf.strings.regex_full_match(filename, ".*_mr")
    handedness = tf.cond(
        is_left_hand,
        lambda: tf.constant([1.0, 0.0], dtype=tf.float32),  # left hand [1, 0]
        lambda: tf.constant([0.0, 1.0], dtype=tf.float32)   # right hand [0, 1]
    )
    
    # hand presence confidence (all samples have hands, so set to 1.0)
    presence_score = tf.constant(1.0, dtype=tf.float32)
    
    # world coordinate system landmarks (use same values as landmarks, but scaled)
    world_landmarks = tf.identity(landmarks)  # create a copy
    
    # process image and coordinates
    processed_image, processed_landmarks = process_image_and_coords(image, landmarks)
    
    # return all needed information
    return (
        processed_image,
        {
            'landmarks': processed_landmarks,
            'handedness': handedness,
            'presence_score': presence_score,
            'world_landmarks': world_landmarks
        }
    )

@register_keras_serializable()
def custom_loss(y_true, y_pred):
    """high precision loss function"""
    y_true_reshaped = tf.reshape(y_true, [-1, 21, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 21, 3])
    
    # separate coordinates
    true_xy = y_true_reshaped[..., :2]
    pred_xy = y_pred_reshaped[..., :2]
    true_z = y_true_reshaped[..., 2:3]
    pred_z = y_pred_reshaped[..., 2:3]
    
    # more detailed point weights
    point_weights = tf.ones([21])
    important_points = {
        2: 1.0,
        4: 1.0,
    }
    
    for idx, weight in important_points.items():
        point_weights = tf.tensor_scatter_nd_update(
            point_weights, [[idx]], [weight]
        )
    
    point_weights = tf.expand_dims(point_weights, 0)
    point_weights = tf.tile(point_weights, [tf.shape(y_true_reshaped)[0], 1])
    point_weights = tf.expand_dims(point_weights, -1)
    
    # use Huber loss instead of MSE to better handle outliers
    def huber_loss(y_true, y_pred, delta=0.5):  # reduce delta value to make loss more sensitive to large errors
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return 0.5 * quadratic * quadratic + delta * linear
    
    # calculate base loss
    xy_loss = tf.reduce_mean(point_weights * tf.reduce_sum(huber_loss(true_xy, pred_xy), axis=2, keepdims=True))
    z_loss = tf.reduce_mean(point_weights * huber_loss(true_z, pred_z))
    
    
    total_loss = (
        xy_loss +
        0.5 * z_loss
    )
    
    return total_loss

@register_keras_serializable()
def custom_metric(y_true, y_pred):
    """improved evaluation metric, evaluate xy and z accuracy separately"""
    # reshape to (batch_size, 21, 3)
    y_true_reshaped = tf.reshape(y_true, [-1, 21, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 21, 3])
    
    # important points list
    important_points = [2, 4]
    
    # extract important points coordinates
    y_true_important = tf.gather(y_true_reshaped, important_points, axis=1)
    y_pred_important = tf.gather(y_pred_reshaped, important_points, axis=1)
    
    # calculate xy and z errors separately
    xy_error = tf.norm(y_true_important[..., :2] - y_pred_important[..., :2], axis=2)
    z_error = tf.abs(y_true_important[..., 2] - y_pred_important[..., 2])
    
    # total error (give z axis higher weight)
    total_error = xy_error + 2.0 * z_error
    
    return tf.reduce_mean(total_error)


train_size = 0  # training set size
total_samples = 0  # total number of samples
BATCH_SIZE = 2 

def prepare_datasets():
    """prepare training and validation datasets"""
    global train_size, total_samples  # declare use of global variables

    # load data
    tfrecord_path = str(Path(__file__).parent / 'training_data.tfrecord')
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    # calculate dataset size
    total_samples = sum(1 for _ in dataset)
    print(f"total number of samples: {total_samples}")
    
    # reset dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # print the shape of the dataset and the first element
    print(f"dataset: {dataset}")
    print(f"first element: {next(iter(dataset))}")
    
    # calculate training and validation set sizes
    train_size = int(0.5 * total_samples)  # 50% for training
    validation_size = total_samples - train_size  # remaining 50% for validation
    
    # split dataset
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # shuffle, repeat, and batch training dataset
    train_dataset = train_dataset.shuffle(buffer_size=10)
    train_dataset = train_dataset.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # batch validation dataset
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    print(f"training set size: {train_size}")
    print(f"validation set size: {validation_size}")
    print(f"batch size: {BATCH_SIZE}")
    print(f"training steps per epoch: {math.ceil(train_size / BATCH_SIZE)}")
    print(f"validation steps per epoch: {math.ceil(validation_size / BATCH_SIZE)}")
    
    return train_dataset, val_dataset

def plot_training_history(history1, history2=None):
    """simplified training history plotting function, only showing key point prediction metrics"""
    plt.figure(figsize=(15, 5))
    
    # plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(history1.history['loss'], label='Stage 1 Total Loss')
    plt.plot(history1.history['val_loss'], label='Stage 1 Val Total Loss')
    if history2:
        plt.plot(range(len(history1.history['loss']), 
                      len(history1.history['loss']) + len(history2.history['loss'])),
                history2.history['loss'], label='Stage 2 Total Loss')
        plt.plot(range(len(history1.history['val_loss']),
                      len(history1.history['val_loss']) + len(history2.history['val_loss'])),
                history2.history['val_loss'], label='Stage 2 Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # plot landmarks MAE
    plt.subplot(1, 2, 2)
    plt.plot(history1.history['landmarks_mae'], label='Stage 1 Landmarks MAE')
    plt.plot(history1.history['val_landmarks_mae'], label='Stage 1 Val Landmarks MAE')
    if history2:
        plt.plot(range(len(history1.history['landmarks_mae']),
                      len(history1.history['landmarks_mae']) + len(history2.history['landmarks_mae'])),
                history2.history['landmarks_mae'], label='Stage 2 Landmarks MAE')
        plt.plot(range(len(history1.history['val_landmarks_mae']),
                      len(history1.history['val_landmarks_mae']) + len(history2.history['val_landmarks_mae'])),
                history2.history['val_landmarks_mae'], label='Stage 2 Val Landmarks MAE')
    plt.title('Landmarks MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    model = create_mediapipe_model()
    train_dataset, val_dataset = prepare_datasets()
    
    steps_per_epoch = math.ceil(train_size / BATCH_SIZE)
    validation_steps = math.ceil((total_samples - train_size) / BATCH_SIZE)
    
    # first stage training
    print("start first stage training...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss={
            'landmarks': custom_loss,
            'handedness': 'categorical_crossentropy',
            'presence_score': 'binary_crossentropy',
            'world_landmarks': 'mse'
        },
        loss_weights={
            'landmarks': 1.0,
            'handedness': 0.001,
            'presence_score': 0.001,
            'world_landmarks': 0.002
        },
        metrics={
            'landmarks': [custom_metric, 'mae'],
            'handedness': ['accuracy'],
            'presence_score': ['accuracy'],
            'world_landmarks': ['mae']
        }
    )

    # use staged cosine decay learning rate
    initial_learning_rate = 0.001
    
    def staged_cosine_decay(epoch):
        if epoch < 100:  
            return initial_learning_rate * (epoch / 100)
        elif epoch < 600:  
            progress = (epoch - 100) / 500
            return initial_learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        else:  
            progress = (epoch - 600) / 400
            return initial_learning_rate * 0.1 * (1 + np.cos(np.pi * progress))

    lr_callback = tf.keras.callbacks.LearningRateScheduler(staged_cosine_decay)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'hand_landmark_model.h5',
            monitor='val_custom_metric',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_custom_metric',
            patience=10,  # increase early stopping patience
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        lr_callback
    ]
    
    # first stage training
    history1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=15,  # increase epoch number
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # second stage: fine-tuning
    print("\nstart second stage fine-tuning...")
    num_layers = len(model.layers)

    # unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss={
            'landmarks': custom_loss,
            'handedness': 'categorical_crossentropy',
            'presence_score': 'binary_crossentropy',
            'world_landmarks': 'mse'
        },
        loss_weights={
            'landmarks': 1.0,
            'handedness': 0.001,
            'presence_score': 0.001,
            'world_landmarks': 0.002
        },
        metrics={
            'landmarks': [custom_metric, 'mae'],
            'handedness': ['accuracy'],
            'presence_score': ['accuracy'],
            'world_landmarks': ['mae']
        }
    )
    # second stage use staged cosine decay
    def fine_tune_staged_decay(epoch):
        initial_lr = 1e-5
        if epoch < 50:  # warmup
            return initial_lr * (epoch / 50)
        elif epoch < 300:  # first cosine cycle
            progress = (epoch - 50) / 250
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        else:  # second cosine cycle
            progress = (epoch - 300) / 200
            return initial_lr * 0.1 * (1 + np.cos(np.pi * progress))

    callbacks[-1] = tf.keras.callbacks.LearningRateScheduler(fine_tune_staged_decay)
    callbacks[1].patience = 8  # increase early stopping patience
    callbacks[2].patience = 5
    
    history2 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history1, history2

if __name__ == "__main__":
    try:
        model, history1, history2 = train_model()
        
        # plot training history
        plot_training_history(history1, history2)
        
        # save model
        model.save('hand_landmark_model.keras')
        try:
            model.save('hand_landmark_model.h5')
        except Exception as e:
            print(f"failed to save h5 format: {e}")
        
        print("training completed, model saved")
        
    except Exception as e:
        print(f"training process error: {e}")
        import traceback
        traceback.print_exc() 