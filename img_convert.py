import nibabel as nib
import os
import numpy as np
np.set_printoptions(threshold=np.nan)

from nibabel.testing import data_path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(precision=2, suppress=True)
brats_seg='/media/lele/DATA/brain/Brats17TrainingData/HGG5/Brats17_TCIA_469_1/Brats17_TCIA_469_1_seg.nii.gz'
brats_ni1 = '/media/lele/DATA/brain/Brats17TrainingData/HGG5/Brats17_TCIA_469_1/deep-brats17.D500.f.p13.c3c3c3c3c3.n32n32n32n32n32.d256.e4.pad_valid.test.nii.gz'
# brats_ni1 = os.path.join(data_path2, 'Brats17_2013_24_1_flair.nii.gz')
=======
#data_path2='/Users/chenlele/Project/brain/data/Brats17TrainingData/LGG/Brats17_2013_24_1'
# data_path2='/mnt/disk0/dat/lchen63/brain/data/Brats17TrainingData/LGG/Brats17_2013_24_1'


# brats_ni1 = os.path.join(data_path2, 'Brats17_2013_24_1_flair.nii.gz')
n1_brimg = nib.load(brats_ni1)
n1_brimg

n1_header = n1_brimg.header
print(n1_header) 
print '----------'
image1=n1_brimg.get_data()
print type(image1)
print image1.shape
image2 = image1[:,:,100]
print image2
find  = 0
# print image2



plt.imshow(image1[:,:,120], cmap='gray')
plt.show()
# print image1[43:-43,43:-43,-25].shape
# brats_t11 = os.path.join(data_path2, 'Brats17_2013_24_1_t1.nii.gz')
# n1_t1img = nib.load(brats_t11)
# imaget1=n1_t1img.get_data()
# plt.imshow(imaget1[:,:,100], cmap='gray')
# plt.show()

# brats_t1ce = os.path.join(data_path2, 'Brats17_2013_24_1_t1ce.nii.gz')
# n1_t1ceimg = nib.load(brats_t1ce)
# imaget1ce=n1_t1ceimg.get_data()
# plt.imshow(imaget1ce[48:192,48:192,100], cmap='gray')
# plt.show()

# brats_t2 = os.path.join(data_path2, 'Brats17_2013_24_1_t2.nii.gz')
# n1_t2img = nib.load(brats_t2)
# imaget2=n1_t2img.get_data()
# plt.imshow(imaget1ce[48:192,48:192,100], cmap='gray')
# plt.show()


# brats_seg = os.path.join(data_path2, 'Brats17_2013_24_1_seg.nii.gz')
n1_segimg = nib.load(brats_seg)
imageseg=n1_segimg.get_data()
# print  imageseg[:,:,100]
plt.imshow(imageseg[:,:,-95], cmap='gray')
=======
# n1_segimg = nib.load(brats_seg)
# imageseg=n1_segimg.get_data()
# print  imageseg[:,:,100]
# plt.imshow(imageseg[:,:,100], cmap='gray')

# plt.show()











