# from __future__ import print_function
import argparse
import os
from time import strftime
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, Input, concatenate, Reshape, Lambda, Permute
from keras.layers.recurrent import LSTM
from nibabel import load as load_nii
from utils import color_codes, fold_train_test_val, get_biggest_region
from itertools import izip
from data_creation import load_patch_batch_train, get_cnn_centers
from data_creation import load_patch_batch_generator_test
from data_manipulation.generate_features import get_mask_voxels
from data_manipulation.metrics import dsc_seg
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.convolutional import Conv3D, Conv3DTranspose, UpSampling3D
from keras.layers.pooling import AveragePooling3D
# from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
import loss_functions as lf

from subpixel import SubPixelUpscaling
os.environ["CUDA_VISIBLE_DEVICES"] = "1"





def parse_inputs():
    # I decided to separate this function, for easier acces to the command line parameters
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
    parser.add_argument('-f', '--folder', dest='dir_name', default='/media/yue/Data/spie/Brats17TrainingData/HGG')
    parser.add_argument('-F', '--n-fold', dest='folds', type=int, default=5)
    parser.add_argument('-i', '--patch-width', dest='patch_width', type=int, default=13)
    parser.add_argument('-k', '--kernel-size', dest='conv_width', nargs='+', type=int, default=3)
    parser.add_argument('-c', '--conv-blocks', dest='conv_blocks', type=int, default=5)
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=16)
    parser.add_argument('-p', '--pred-size', dest='pred_size', type=int, default=1)
    parser.add_argument('-d', '--dense-size', dest='dense_size', type=int, default=256)
    parser.add_argument('-D', '--down-factor', dest='dfactor', type=int, default=500)
    parser.add_argument('-s', '--sequntial', dest='sequntial', type=bool, default=False)
    parser.add_argument('-n', '--num-filters', action='store', dest='n_filters', nargs='+', type=int, default=[32])
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=1)
    parser.add_argument('-q', '--queue', action='store', dest='queue', type=int, default=10)
    parser.add_argument('-u', '--unbalanced', action='store_false', dest='balanced', default=True)
    parser.add_argument('-bb', '--binary', action='store', dest='binary', default=False)
    parser.add_argument('-dd', '--dense', action='store', dest='dense', default=False)
    parser.add_argument('-C', '--continue-training', dest='continue', default=False)
    parser.add_argument('--preload', action='store_true', dest='preload', default=False)
    parser.add_argument('--padding', action='store', dest='padding', default='valid')
    parser.add_argument('--no-flair', action='store_false', dest='use_flair', default=True)
    parser.add_argument('--no-t1', action='store_false', dest='use_t1', default=True)
    parser.add_argument('--no-t1ce', action='store_false', dest='use_t1ce', default=True)
    parser.add_argument('--no-t2', action='store_false', dest='use_t2', default=True)
    parser.add_argument('--flair', action='store', dest='flair', default='_flair.nii.gz')
    parser.add_argument('--t1', action='store', dest='t1', default='_t1.nii.gz')
    parser.add_argument('--t1ce', action='store', dest='t1ce', default='_t1ce.nii.gz')
    parser.add_argument('--t2', action='store', dest='t2', default='_t2.nii.gz')
    parser.add_argument('--labels', action='store', dest='labels', default='_seg.nii.gz')
    parser.add_argument('-m', '--multi-channel', action='store_true', dest='multi', default=False)
    return vars(parser.parse_args())


def list_directories(path):
    return filter(os.path.isdir, [os.path.join(path, f) for f in os.listdir(path)])


def get_names_from_path(options):
    path = options['dir_name']
    patients = []
    with open('train.txt') as f:
        for line in f:
            patients.append(os.path.join(path, line[:-1]))

    train_len = len(patients)
    with open('val.txt') as f:
        for line in f:
            patients.append(os.path.join(path, line[:-1]))

    with open('test.txt') as f:
        for line in f:
            patients.append(os.path.join(path, line[:-1]))

    # patients = sorted(list_directories(path))
    #
    # patients = patients[:6] # multiply of 6

    # Prepare the names
    flair_names = [os.path.join(path, p, p.split('/')[-1] + options['flair'])
                   for p in patients] if options['use_flair'] else None
    t2_names = [os.path.join(path, p, p.split('/')[-1] + options['t2'])
                for p in patients] if options['use_t2'] else None
    t1_names = [os.path.join(path, p, p.split('/')[-1] + options['t1'])
                for p in patients] if options['use_t1'] else None
    t1ce_names = [os.path.join(path, p, p.split('/')[-1] + options['t1ce'])
                  for p in patients] if options['use_t1ce'] else None
    label_names = np.array([os.path.join(path, p, p.split('/')[-1] + options['labels']) for p in patients])
    image_names = np.stack(filter(None, [flair_names, t2_names, t1_names, t1ce_names]), axis=1)

    return image_names, label_names, train_len

def center_model(patch_size, num_classes):
    model = Sequential()
    model.add(Conv3D(64,(3,3,3),strides=1, padding='same',activation= 'relu',data_format = 'channels_first', input_shape=(4, patch_size,patch_size,patch_size)))
    model.add(Conv3D(64,(3,3,3),strides=1, padding='same',activation= 'relu',data_format = 'channels_first'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_first'))

    model.add(Conv3D(128,(3,3,3),strides=1, padding='same',activation= 'relu',data_format = 'channels_first'))
    model.add(Conv3D(128,(3,3,3),strides=1, padding='same',activation= 'relu',data_format = 'channels_first'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2, data_format='channels_first'))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    return model

def main():
    options = parse_inputs()
    c = color_codes()

    # Prepare the net architecture parameters
    dfactor = options['dfactor']
    # Prepare the net hyperparameters
    num_classes = 5
    epochs = options['epochs']
    padding = options['padding']
    patch_width = options['patch_width']
    patch_size = (patch_width, patch_width, patch_width)
    batch_size = options['batch_size']
    pred_size = options['pred_size']
    dense_size = options['dense_size']
    conv_blocks = options['conv_blocks']
    n_filters = options['n_filters']
    continue_training = options['continue']
    filters_list = n_filters if len(n_filters) > 1 else n_filters*conv_blocks
    conv_width = options['conv_width']
    kernel_size_list = conv_width if isinstance(conv_width, list) else [conv_width]*conv_blocks
    balanced = options['balanced']
    # Data loading parameters
    preload = options['preload']
    queue = options['queue']
    # save_path = options['save_path']

    # Prepare the sufix that will be added to the results for the net and images
    path = options['dir_name']
    # filters_s = 'n'.join(['%d' % nf for nf in filters_list])
    # conv_s = 'c'.join(['%d' % cs for cs in kernel_size_list])
    # s_s = '.s' if sequential else '.f'
    # ub_s = '.ub' if not balanced else ''
    # params_s = (ub_s, dfactor, s_s, patch_width, conv_s, filters_s, dense_size, epochs, padding)
    # sufix = '%s.D%d%s.p%d.c%s.n%s.d%d.e%d.pad_%s.' % params_s
    n_channels = np.count_nonzero([
        options['use_flair'],
        options['use_t2'],
        options['use_t1'],
        options['use_t1ce']]
    )

    print(c['c'] + '[' + strftime("%H:%M:%S") + '] ')
    # N-fold cross validation main loop (we'll do 2 training iterations with testing for each patient)
    data_names, label_names, train_len = get_names_from_path(options)
    folds = options['folds']
    # (train_data, train_labels, val_data, val_labels, test_data, test_labels) = fold_train_test_val(data_names, label_names,val_data=0.25)
    datas = fold_train_test_val(data_names, label_names,val_data=train_len)
    train_data, train_labels, val_data, val_labels, test_data, test_labels= datas[0], datas[1], datas[2], datas[3], datas[4], datas[5]
    dsc_results = list()
    dsc_results = list()
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']  ' + c['nc'] + c['g'] +
          'Number of training/validation/testing images (%d=%d/%d=%d/%d)'
          % (len(train_data), len(train_labels), len(val_data), len(val_labels), len(test_data)) + c['nc'])
    # Prepare the data relevant to the leave-one-out (subtract the patient from the dataset and set the path)
    
    net_name = os.path.join(path, 'center.hdf5')
    # First we check that we did not train for that patient, in order to save time
   
    print '==============================================================='
    # NET definition using Keras
    train_centers = get_cnn_centers(train_data[:, 0], train_labels, balanced=balanced)
    val_centers = get_cnn_centers(val_data[:, 0], val_labels, balanced=balanced)
    train_samples = len(train_centers)/dfactor
    val_samples = len(val_centers) / dfactor
    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' + c['g'] + 'Creating and compiling the model ' +
          c['b'] + '(%d samples)' % train_samples + c['nc'])
    train_steps_per_epoch = -(-train_samples/batch_size)
    val_steps_per_epoch = -(-val_samples / batch_size)
    # input_shape = (n_channels,) + patch_size

    # This architecture is based on the functional Keras API to introduce 3 output paths:
    # - Whole tumor segmentation
    # - Core segmentation (including whole tumor)
    # - Whole segmentation (tumor, core and enhancing parts)
    # The idea is to let the network work on the three parts to improve the multiclass segmentation.

    # net = Model(inputs=merged_inputs, outputs=[tumor])
    
    net=center_model(patch_size=patch_size, num_classes=num_classes)


    # net_name_before =  os.path.join(path,'baseline-brats2017.fold0.D500.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e1.pad_valid.mdl')
    # net = keras.models.load_model(net_name_before)
    if continue_training:
        net.load_weights(net_name)
        print 'weights loaded'

    net.compile(optimizer='sgd', loss=lf.segmentation_loss, metrics=['accuracy'])

    print(c['c'] + '[' + strftime("%H:%M:%S") + ']    ' +
          c['g'] + 'Training the model with a generator for ' +
          c['b'] + '(%d parameters)' % net.count_params() + c['nc'])
    # print(net.summary())



    net.fit_generator(
            generator=load_patch_batch_train(
            image_names=train_data,
            label_names=train_labels,
            centers=train_centers,
            batch_size=batch_size,
            pred_size=pred_size,
            size=patch_size,
            # fc_shape = patch_size,
            nlabels=num_classes,
            dfactor=dfactor,
            preload=preload,
            binary = options['binary'],
            split= False,
            datatype=np.float32
        ),
      
        steps_per_epoch=train_steps_per_epoch,
        workers=queue,
        max_q_size=queue,
        epochs=epochs
    )
    net.save(net_name)


if __name__ == '__main__':
    main()








