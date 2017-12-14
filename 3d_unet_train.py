import os
import glob
import pickle
import tables
from model import unet_model_3d
import os
from random import shuffle
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.models import load_model
import math
from functools import partial

K.set_image_dim_ordering('tf')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def generate_list(data_type = None):
    path = '/mnt/disk0/dat/lchen63/brain/newdata/'
    if data_type == 'train':
        train = path + 'train/'
        t1ce = train + 't1ce/'
        files = os.listdir(t1ce)
        data_list = []
        for file in files:
            if file[0] =='.':
                continue
            #filename_without_suffix = ''.join(file.split('_')[:-1])
            filename_without_suffix = file[:-9]
            data_list.append(filename_without_suffix)
        return data_list
    elif data_type == 'validation':
        validation = path + 'validation/'
        t1ce = validation + 't1ce/'
        files = os.listdir(t1ce)
        data_list = []
        for file in files:
            if file[0] =='.':
                continue
            #filename_without_suffix = ''.join(file.split('_')[:-1])
            filename_without_suffix = file[:-9]
            data_list.append(filename_without_suffix)
        return data_list
def data_augment(filename_x = None, filename_y = None,augment = False):
    x = np.load(filename_x)
    y = np.load(filename_y)
    if augment == False:
        return x,y
    else:
        #pass
        return x,y
def data_generator(data_list = None, data_type = None, batch_size= 1, augment = False):
    if data_type == 'train':
        x_path = '/mnt/disk0/dat/lchen63/brain/newdata/train/t1ce/'
        y_path = '/mnt/disk0/dat/lchen63/brain/newdata/train/seg/'
    elif data_type == 'validation':
        x_path = '/mnt/disk0/dat/lchen63/brain/newdata/validation/t1ce/'
        y_path = '/mnt/disk0/dat/lchen63/brain/newdata/validation/seg/'
    data_list = data_list
    num_batches = len(data_list) / batch_size
    print '\nby generator: %d batches per epoch' % num_batches
    while 1:
        shuffle(data_list)
        for b in range(num_batches):
            xs = []
            ys =[]
            offset = b * batch_size
            for bs in range(batch_size):
                item = data_list[offset + bs]
                filename_x = x_path + item + '_t1ce.npy'
                filename_y = y_path + item + '_seg.npy'
                x,y = data_augment(filename_x,filename_y,augment)
                xs.append(x)
                ys.append(y)
            yield (np.array(xs),np.array(ys))

def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)
class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    result =  initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))
    return result
def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):
    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    logger = CSVLogger(os.path.join(logging_dir, "training.log"))
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay,
                                              initial_lrate=initial_learning_rate,
                                              drop=learning_rate_drop,
                                              epochs_drop=learning_rate_epochs))
    return [model_checkpoint, logger, history, scheduler]


def main(overwrite=False):
    input_shape = (192,192,64,1)
    downsize_nb_filters_factor = 1
    pool_size = (2,2,2)
    n_labels = 4 #including background
    initial_learning_rate = 0.00001
    batch_size = 1
    n_epochs = 50
    learning_rate_drop = 0.5
    learning_rate_epochs = 10
    model_file = '/mnt/disk0/dat/lchen63/brain/model/3d_unet.hdf5'




    # get training and testing generators
    train_list = generate_list('train')
    train_generator= data_generator(train_list,'train',batch_size,False)
    steps_per_epoch = len(train_list) / batch_size

    validation_list = generate_list('validation')
    validation_generator= data_generator(validation_list,'validation',batch_size,False)
    validation_steps = len(validation_list) / batch_size


    # instantiate new model
    model = unet_model_3d(input_shape=input_shape,
                          downsize_filters_factor=downsize_nb_filters_factor,
                          pool_size=pool_size, n_labels=n_labels,
                          initial_learning_rate=initial_learning_rate)
    print 'load model successfully!'


    model.fit_generator(generator=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        pickle_safe=True,
                        callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs))
    model.save(model_file)


if __name__ == "__main__":
    main(overwrite=False)
