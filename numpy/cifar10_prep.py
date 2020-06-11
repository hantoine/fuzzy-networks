
def load_cifar10():
    import numpy as np
    import tensorflow as tf

    (images_train, labels_train), (images_test, labels_test) = \
        tf.keras.datasets.cifar10.load_data()

    def normalize(array):
        return array.astype('float32') / 255.0

    def flatten(threeDarray):
        # Originally 32x32x3 for rgb channels
        return [np.reshape(x, 3072) for x in threeDarray]

    def one_hot(array, n_classes=10):
        return np.eye(n_classes)[array].reshape(len(array), n_classes)

    images_train = normalize(images_train)
    images_test = normalize(images_test)

    labels_train = one_hot(labels_train)
    labels_test = one_hot(labels_test)

    return images_train, images_test, labels_train, labels_test
