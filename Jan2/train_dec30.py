import numpy as np
import tf_models
from sklearn.preprocessing import scale
import argparse
import tensorflow as tf
from nibabel import load as load_nii
import os
from tensorflow.contrib.keras.python.keras.backend import learning_phase
NUM_EPOCHS = 2
LOAD_PATH = 'phase2/BraTS2ScaleDenseNetP2-1'
SAVE_PATH = 'BraTS2ScaleDenseNetP3'
# OFFSET_W = 8
# OFFSET_H = 8
# OFFSET_C = 8
PSIZE = 12
HSIZE = 38
WSIZE = 38
CSIZE = 38
BATCH_SIZE = 4
# batches_h, batches_w, batches_c = (224-HSIZE)/OFFSET_H+1, (224-WSIZE)/OFFSET_W+1, (152 - CSIZE)/OFFSET_C+1
# iter_per_epoch = batches_h*batches_w*batches_c

def parse_inputs():

    parser = argparse.ArgumentParser(description='yyy')
    parser.add_argument('-r', '--root-path', dest='root_path', default='/media/yue/Data/spie/Brats17TrainingData/HGG')
    parser.add_argument('-m', '--model-path', dest='model_path',
                        default='BraTS2ScaleDenseNetP3-0')
    parser.add_argument('-ow', '--offset-width', dest='offset_w', type=int, default=12)
    parser.add_argument('-oh', '--offset-height', dest='offset_h', type=int, default=12)
    parser.add_argument('-oc', '--offset-channel', dest='offset_c', nargs='+', type=int, default=12)
    parser.add_argument('-ws', '--width-size', dest='wsize', type=int, default=38)
    parser.add_argument('-hs', '--height-size', dest='hsize', type=int, default=38)
    parser.add_argument('-cs', '--channel-size', dest='csize', type=int, default=38)
    parser.add_argument('-ps', '--pred-size', dest='psize', type=int, default=12)

    return vars(parser.parse_args())

options = parse_inputs()

def acc_tf(y_pred, y_true):
    correct_prediction = tf.equal(tf.cast(tf.argmax(y_pred, -1), tf.int32), y_true)
    return 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def get_patches_3d(data, labels, centers, hsize, wsize, csize, psize, preprocess=True):
    """

    :param data: 4D nparray (h, w, c, ?)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """
    patches_x, patches_y = [], []
    offset_p = (hsize - psize) / 2
    for i in range(len(centers[0])):
        h, w, c = centers[0, i], centers[1, i], centers[2, i]
        h_beg = min(max(0, h - hsize / 2), 240 - hsize)
        w_beg = min(max(0, w - wsize / 2), 240 - wsize)
        c_beg = min(max(0, c - csize / 2), 155 - csize)
        ph_beg = h_beg + offset_p
        pw_beg = w_beg + offset_p
        pc_beg = c_beg + offset_p
        vox = data[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize, :]
        vox_labels = labels[ph_beg:ph_beg + psize, pw_beg:pw_beg + psize, pc_beg:pc_beg + psize]
        patches_x.append(vox)
        patches_y.append(vox_labels)
    return np.array(patches_x), np.array(patches_y)


def positive_ratio(x):
    return float(np.sum(np.greater(x, 0))) / np.prod(x.shape)


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def segmentation_loss(y_true, y_pred, n_classes):
    y_true = tf.one_hot(y_true, depth=n_classes, axis=-1)
    y_true = tf.reshape(y_true, (-1, n_classes))
    y_pred = tf.reshape(y_pred, (-1, n_classes))
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                                  logits=y_pred))


def vox_preprocess(vox):
    vox_shape = vox.shape
    vox = np.reshape(vox, (-1, vox_shape[-1]))
    vox = scale(vox, axis=0)
    return np.reshape(vox, vox_shape)


def one_hot(y, num_classes):
    y_ = np.zeros([len(y), num_classes])
    y_[np.arange(len(y)), y] = 1
    return y_


def dice_coef_np(y_true, y_pred, num_classes):
    """

    :param y_true: sparse labels
    :param y_pred: sparse labels
    :param num_classes: number of classes
    :return:
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    y_true = y_true.flatten()
    y_true = one_hot(y_true, num_classes)
    y_pred = y_pred.flatten()
    y_pred = one_hot(y_pred, num_classes)
    intersection = np.sum(y_true * y_pred, axis=0)
    return (2. * intersection) / (np.sum(y_true, axis=0) + np.sum(y_pred, axis=0))


def vox_generator(all_files, n_pos, n_neg):
    path = options['root_path']
    while 1:
        for file in all_files:
            flair = load_nii(os.path.join(path, file, file + '_flair.nii.gz')).get_data()
            t2 = load_nii(os.path.join(path, file, file + '_t2.nii.gz')).get_data()
            t1 = load_nii(os.path.join(path, file, file + '_t1.nii.gz')).get_data()
            t1ce = load_nii(os.path.join(path, file, file + '_t1ce.nii.gz')).get_data()

            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = load_nii(os.path.join(path, file, file+'_seg.nii.gz')).get_data()

            foreground = np.array(np.where(labels > 0))
            background = np.array(np.where((labels == 0) & (flair > 0)))

            # n_pos = int(foreground.shape[1] * discount)
            foreground = foreground[:, np.random.permutation(foreground.shape[1])[:n_pos]]
            background = background[:, np.random.permutation(background.shape[1])[:n_neg]]

            centers = np.concatenate((foreground, background), axis=1)
            centers = centers[:, np.random.permutation(n_neg+n_pos)]

            yield data_norm, labels, centers


def train(continue_training):
    files = []
    with open('train.txt') as f:
        for line in f:
            files.append(line[:-1])
    print '%d training samples' % len(files)

    data_node = tf.placeholder(dtype=tf.float32, shape=(None, HSIZE, WSIZE, CSIZE, 4))
    label_node = tf.placeholder(dtype=tf.int32, shape=(None, PSIZE, PSIZE, PSIZE))
    logits = tf_models.BraTS2ScaleDenseNet(input=data_node, num_labels=5)
    loss = segmentation_loss(y_true=label_node, y_pred=logits, n_classes=5)
    acc_batch = acc_tf(y_pred=logits, y_true=label_node)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss)

    saver = tf.train.Saver(max_to_keep=15)
    data_gen_train = vox_generator(all_files=files, n_pos=300, n_neg=100)

    with tf.Session() as sess:
        if continue_training:
            saver.restore(sess, LOAD_PATH)
        else:
            sess.run(tf.global_variables_initializer())
        for ei in range(NUM_EPOCHS):
            for pi in range(len(files)):
                acc_pi, loss_pi = [], []
                data, labels, centers = data_gen_train.next()
                n_batches = int(np.ceil(float(centers.shape[1]) / BATCH_SIZE))
                for nb in range(n_batches):
                    offset_batch = min(nb * BATCH_SIZE, centers.shape[1] - BATCH_SIZE)
                    data_batch, label_batch = get_patches_3d(data, labels, centers[:, offset_batch:offset_batch + BATCH_SIZE], HSIZE, WSIZE, CSIZE, PSIZE, False)
                    _, l, acc = sess.run(fetches=[optimizer, loss, acc_batch], feed_dict={data_node:data_batch,
                                                                          label_node:label_batch,
                                                                          learning_phase(): 1})
                    acc_pi.append(acc)
                    loss_pi.append(l)
                    n_pos = len(np.where(label_batch > 0)[0])
                    print 'epoch-patient: %d, %d, iter: %d-%d, p%%: %.4f, loss: %.4f, acc: %.2f%%' % \
                          (ei + 1, pi + 1, nb + 1, n_batches, n_pos/float(np.prod(label_batch.shape)), l, acc)

                print 'patient loss: %.4f, patient acc: %.4f' % (np.mean(loss_pi), np.mean(acc_pi))

            saver.save(sess, SAVE_PATH, global_step=ei)
            print 'model saved'

if __name__ == '__main__':
    train(continue_training=True)