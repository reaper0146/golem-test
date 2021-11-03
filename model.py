import tensorflow as tf
import numpy as np
import glob

DATASET_PATH = '/golem/dataset/california_housing_train'



def get_keras_model():
    
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax(),
    ])


def get_compiled_model():

    model = get_keras_model()
    optimizer = 'adam'
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


def load_model_from_file(file_path):

    return tf.keras.models.load_model(file_path)


def load_dataset(batch_size):
    

    (X_train, y_train), (X_test, y_test) = tf.data.Dataset(DATASET_PATH)
    train_length = len(X_train)
    test_length = len(X_test)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (X_test, y_test)).batch(batch_size)
    return train_dataset, test_dataset, train_length, test_length


def avg_weights(all_client_weights, weights=None):
    
    return np.average(np.array(all_client_weights), axis=0, weights=weights)


def get_client_model_weights(worker_model_folder, round_num):
   
    client_weights = []
    for model_weights in glob.glob(f'{worker_model_folder}/round_{round_num}_worker_*[0-9].h5'):
        temp_model = load_model_from_file(f'{model_weights}')
        client_weights.append(temp_model.get_weights())
    return client_weights
