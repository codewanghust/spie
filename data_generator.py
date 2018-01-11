import os
from random import shuffle
import numpy as np
import subprocess
from nipype.interfaces.ants import N4BiasFieldCorrection

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
def n4normalization(path, n_dims=3, n_iters='[20,20,10,5]'):
    '''
    INPUT:  (1) filepath 'path': path to mha T1 or T1c file
            (2) directory 'parent_dir': parent directory to mha file
    OUTPUT: writes n4itk normalized image to parent_dir under orig_filename_n.mha
    '''
    output_fn = path[:-7] + '__n.nii.gz'
    # n4_norm(output_fn,n_dim,n_iters)
    subprocess.call('python n4_bias_correction.py ' + path + ' ' + str(n_dims) + ' ' + n_iters + ' ' + output_fn, shell = True)
    # N4BiasFieldCorrection(output_image=sys.argv[4])
    # run n4_bias_correction.py path n_dim n_iters output_fn




n4normalization('/media/lele/DATA/brain/Brats17TrainingData/HGG/Brats17_2013_10_1/Brats17_2013_10_1_t1ce.nii.gz')


# import copy
# from nipype.interfaces.ants import N4BiasFieldCorrection
# n4 = N4BiasFieldCorrection()
# n4.inputs.dimension = 3
# n4.inputs.input_image = '/media/lele/DATA/brain/Brats17TrainingData/HGG/Brats17_2013_10_1/Brats17_2013_10_1_t1ce.nii.gz'
# n4.inputs.bspline_fitting_distance = 300
# n4.inputs.shrink_factor = 3
# n4.inputs.n_iterations = [50,50,30,20]
# n4.cmdline
# 'N4BiasFieldCorrection --bspline-fitting [ 300 ] -d 3 --input-image structural.nii --convergence [ 50x50x30x20 ] --output /media/lele/DATA/brain/Brats17TrainingData/HGG/Brats17_2013_10_1/Brats17_2013_10_1_t1ce__correct.nii.gz --shrink-factor 3'

# print 'hhhehe'