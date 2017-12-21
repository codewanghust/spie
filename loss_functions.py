import tensorflow as tf
import keras.backend as K


def segmentation_loss(y_true, y_pred):
    n_classes = K.int_shape(y_pred)[-1]
    # y_true = K.cast(y_true, 'int32')
    # y_true = K.one_hot(y_true, num_classes=n_classes)
    y_true = tf.reshape(y_true, (-1, n_classes))
    y_pred = tf.reshape(y_pred, (-1, n_classes))
    y_pred = tf.nn.softmax(y_pred)
    return K.mean(K.categorical_crossentropy(output=y_pred, target=y_true))