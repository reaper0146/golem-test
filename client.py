import tensorflow as tf
import json
from collections import namedtuple


DATASET_PATH = '/golem/dataset/california_housing_train'
USE_FILE = '/golem/work/using.json'


def get_train_dataset(start, end, batch_size):
    (X_train, y_train), (X_test, y_test) = tf.data.Dataset(DATASET_PATH)
    X_train, X_test = X_train / 255.0, X_test / 255.0
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train[start:end], y_train[start:end])).batch(batch_size)
    return train_dataset


def load_model_from_file(file_path):
    return tf.keras.models.load_model(file_path)


def main():
    using = json.load(open(USE_FILE, 'r'))
    using = namedtuple('RoundSpecs', using.keys())(*using.values())
    training_dataset = get_train_dataset(
        using.start, using.end, using.batch_size)
    model = load_model_from_file(using.model_path)
    train_history = model.fit(training_dataset, epochs=using.epochs)
    model.save(f'/golem/output/model_round_{using.global_round}_{using.node_number}.h5')
    with open(f'/golem/output/log_round_{using.global_round}_{using.node_number}.json', 'w') as log_file:
        log_file.write(json.dumps(train_history.history))
    print('Training Done. BBye!')


if __name__ == "__main__":
    main()
