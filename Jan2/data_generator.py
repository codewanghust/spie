import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

MEAN_FLAIR = 41.4745
MEAN_T1 = 70.9409
MEAN_T2 = 70.9558
MEAN_T1CE = 76.3757
STD_FLAIR = 118.5504
STD_T1 = 236.5435
STD_T2 = 217.5261
STD_T1ce = 255.0620


def to_nparray(list):
    ret = None
    for item in list:
        if ret is None:
            ret = item
        else:
            ret = np.concatenate((ret, item))
    return ret


def get_patches_3d(data, labels, centers, hsize, wsize, csize, preprocess=True):
    """

    :param data: 4D nparray (h, w, c, ?)
    :param centers:
    :param hsize:
    :param wsize:
    :param csize:
    :return:
    """
    patches_x, patches_y = [], []
    for i in range(len(centers[0])):
        h, w, c = centers[0][i], centers[1][i], centers[2][i]
        h_beg = min(max(0, h - hsize / 2), 240 - hsize)
        w_beg = min(max(0, w - wsize / 2), 240 - wsize)
        c_beg = min(max(0, c - csize / 2), 155 - csize)
        vox = data[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize, :]
        if preprocess:
            vox_shape = vox.shape
            vox = np.reshape(vox, (-1, vox_shape[-1]))
            vox = scale(vox, axis=0)
            vox = np.reshape(vox, vox_shape)
        vox_labels = labels[h_beg:h_beg + hsize, w_beg:w_beg + wsize, c_beg:c_beg + csize]
        patches_x.append(vox)
        patches_y.append(vox_labels)
    return patches_x, patches_y


def random_centers(idx_list, num_centers):
    ret_list = []
    for idx in idx_list:
        ret_idx_list = []
        perm = np.random.permutation(len(idx[0]))
        perm = perm[:num_centers]
        for axis in idx:
            ret_idx_list.append(axis[perm])
        ret_list.append(ret_idx_list)
    return ret_list


def norm(image):
    image = np.squeeze(image)
    image_nonzero = image[np.nonzero(image)]
    return (image - image_nonzero.mean()) / image_nonzero.std()


def vox_generator_test(all_files):
    path_flair = '/home/yue/Desktop/data/brain_HGG/train/flair/'
    path_t1 = '/home/yue/Desktop/data/brain_HGG/train/t1/'
    path_t2 = '/home/yue/Desktop/data/brain_HGG/train/t2/'
    path_t1ce = '/home/yue/Desktop/data/brain_HGG/train/t1ce/'

    while 1:
        # np.random.shuffle(all_files)
        for file in all_files:

            flair = np.load(path_flair + file + '_flair.npy')
            t1 = np.load(path_t1 + file + '_t1.npy')
            t2 = np.load(path_t2 + file + '_t2.npy')
            t1ce = np.load(path_t1ce + file + '_t1ce.npy')
            data = np.array([flair, t2, t1, t1ce])
            data = np.transpose(data, axes=[1, 2, 3, 0])
            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = np.load('/home/yue/Desktop/data/brain_HGG/train/seg/' + file + '_seg.npy')
            yield data, data_norm, labels


def vox_generator(all_files, n_pos, n_neg):
    path_flair = '/home/yue/Desktop/data/brain_HGG/train/flair/'
    path_t1 = '/home/yue/Desktop/data/brain_HGG/train/t1/'
    path_t2 = '/home/yue/Desktop/data/brain_HGG/train/t2/'
    path_t1ce = '/home/yue/Desktop/data/brain_HGG/train/t1ce/'

    while 1:
        for file in all_files:
            flair = np.load(path_flair + file + '_flair.npy')
            t1 = np.load(path_t1 + file + '_t1.npy')
            t2 = np.load(path_t2 + file + '_t2.npy')
            t1ce = np.load(path_t1ce + file + '_t1ce.npy')
            data_norm = np.array([norm(flair), norm(t2), norm(t1), norm(t1ce)])
            data_norm = np.transpose(data_norm, axes=[1, 2, 3, 0])
            labels = np.load('/home/yue/Desktop/data/brain_HGG/train/seg/' + file + '_seg.npy')

            foreground = np.array(np.where(labels > 0))
            background = np.array(np.where((labels == 0) & (flair > 0)))

            # n_pos = int(foreground.shape[1] * discount)
            foreground = foreground[:, np.random.permutation(foreground.shape[1])[:n_pos]]
            background = background[:, np.random.permutation(background.shape[1])[:n_neg]]

            centers = np.concatenate((foreground, background), axis=1)
            centers = centers[:, np.random.permutation(n_neg+n_pos)]

            yield data_norm, labels, centers





if __name__ == "__main__":
    sv_path = '../imgs/'
    files = []
    with open('../train.txt') as f:
        for line in f:
            files.append(line[:-1])
    data_gen = vox_generator(files, num_centers=20)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    x_all, y_all = data_gen.next()
    print (x_all.shape, y_all.shape)
    # print x_all.shape
    # idx = 57
    # data = x_all[idx]
    # labels = y_all[idx]
    # for i in range(0, 16):
    #     print 'saving %d' % i
    #     f, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5, subplot_kw={'xticks': [], 'yticks': []})
    #     f.set_figheight(10)
    #     f.set_figwidth(20)
    #     ax0.imshow(data[:, :, i, 0], cmap='gray')
    #     ax0.set_title('flair', fontsize=40)
    #     ax1.imshow(data[:, :, i, 1], cmap='gray')
    #     ax1.set_title('t1', fontsize=40)
    #     ax2.imshow(data[:, :, i, 2], cmap='gray')
    #     ax2.set_title('t2', fontsize=40)
    #     ax3.imshow(data[:, :, i, 3], cmap='gray')
    #     ax3.set_title('t1ce', fontsize=40)
    #     ax4.imshow(labels[:, :, i])
    #     ax4.set_title('labels', fontsize=40)
    #     plt.savefig(sv_path + str(i) + '.png')