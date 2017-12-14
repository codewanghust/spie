import os
from random import shuffle
import numpy as np
def generate_list():
	path = '/Users/chenlele/Project/brain/data/'
	train = path + 'train/'
	validation = path + 'validation/'
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

def data_augment(filename_x = None, filename_y = None,augment = False):
	x = np.load(filename_x)
	y = np.load(filename_y)
	if augment == False:
		return x,y
	else:
		#pass
		return x,y
def data_generator(data_list = None, batch_size= 8, augment = False):
	x_path = '/Users/chenlele/Project/brain/data/train/t1ce/'
	y_path = '/Users/chenlele/Project/brain/data/train/seg/'
	data_list = data_list
	num_batches = len(data_list) / batch_size
	print 'by generator: %d batches per epoch' % num_batches
	while 1:
		shuffle(data_list)
		for b in range(num_batches):
			print b
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
data_list = generate_list()
generator= data_generator(data_list,8,False)
xs , ys = generator.next()
print len(xs)
print len(ys)









