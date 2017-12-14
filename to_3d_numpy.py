import nibabel as nib
import os
import numpy as np
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
np.set_printoptions(precision=2, suppress=True)

path = '/mnt/disk0/dat/lchen63/brain/newdata/'
train_HGG_txt = path + 'train_HGG.txt'
validation_HGG_txt = path + 'val_HGG.txt'

train_HGG_reader = open(train_HGG_txt,'r')
val_HGG_reader = open(validation_HGG_txt,'r')

if not os.path.exists(path + 'train'):
	os.mkdir(path + 'train')

if not os.path.exists(path + 'validation'):
	os.mkdir(path + 'validation')

if not os.path.exists(path + 'train/' + 't1'):
	os.mkdir(path + 'train/t1')

if not os.path.exists(path + 'validation/t1'):
	os.mkdir(path + 'validation/t1')

if not os.path.exists(path + 'train/' + 't1ce'):
	os.mkdir(path + 'train/t1ce')

if not os.path.exists(path + 'validation/t1ce'):
	os.mkdir(path + 'validation/t1ce')

if not os.path.exists(path + 'train/' + 't2'):
	os.mkdir(path + 'train/t2')

if not os.path.exists(path + 'validation/t2'):
	os.mkdir(path + 'validation/t2')

if not os.path.exists(path + 'train/' + 'flair'):
	os.mkdir(path + 'train/flair')

if not os.path.exists(path + 'validation/flair'):
	os.mkdir(path + 'validation/flair')

if not os.path.exists(path + 'train/' + 'seg'):
	os.mkdir(path + 'train/seg')

if not os.path.exists(path + 'validation/seg'):
	os.mkdir(path + 'validation/seg')

tensor_size = [192,192,64]

heigh_start = int((240-tensor_size[0])/2)
depth_start = int(155-tensor_size[2])/2
depth_end = depth_start + tensor_size[2]

b = 0
for line in val_HGG_reader:
	line = line[:-1]
	n1_brimg = nib.load(line)
	image=n1_brimg.get_data()
	print image.shape
	
	print '------------------'
	print line
	temp = line.split('/')
	tensor = np.zeros((tensor_size[0],tensor_size[1],tensor_size[2],1),float)
	if 't1' in temp[-1]:
		if 't1ce' not in line:
			outfile = path + 'validation/t1/' + temp[-1][:-7] + '.npy'
			tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
			np.save(outfile, tensor)
		else:
			outfile = path + 'validation/t1ce/' + temp[-1][:-7] + '.npy'
			print outfile
			print '++++'
			b += 1
			tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
			np.save(outfile, tensor)
	elif 't2' in temp[-1]:
		outfile = path + 'validation/t2/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)
	elif 'flair' in temp[-1]:
		outfile = path + 'validation/flair/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)
	elif 'seg' in temp[-1]:

		outfile = path + 'validation/seg/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)

c = 0

for line in train_HGG_reader:
	line = line[:-1]
	n1_brimg = nib.load(line)
	image=n1_brimg.get_data()
	print '------------------'
	print line
	temp = line.split('/')
	tensor = np.zeros((tensor_size[0],tensor_size[1],tensor_size[2],1),float)
	if 't1' in temp[-1]:
		if 't1ce' not in line:
			outfile = path + 'train/t1/' + temp[-1][:-7] + '.npy'
			tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
			np.save(outfile, tensor)
		else:
			outfile = path + 'train/t1ce/' + temp[-1][:-7] + '.npy'
			print outfile
			c += 1
			print '____'
			tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
			np.save(outfile, tensor)
	elif 't2' in temp[-1]:
		outfile = path + 'train/t2/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)
	elif 'flair' in temp[-1]:
		outfile = path + 'train/flair/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)
	elif 'seg' in temp[-1]:
		outfile = path + 'train/seg/' + temp[-1][:-7] + '.npy'
		tensor[:,:,:,0] = image[heigh_start:heigh_start + tensor_size[0],heigh_start:heigh_start + tensor_size[1],depth_start: depth_start +tensor_size[2] ]
		np.save(outfile, tensor)
print b
print c



