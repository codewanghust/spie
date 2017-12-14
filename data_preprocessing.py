 import os
import numpy as np

path = '/Users/chenlele/Project/brain/data/'

train_path = path + 'train/'
val_path = path + 'validation/'


prepross_path = '/Users/chenlele/Project/brain/preprocessed_data/'
if not os.path.exists(prepross_path  ):
	os.mkdir(prepross_path )

if not os.path.exists(prepross_path + 'train/' ):
	os.mkdir(prepross_path + 'train/')

if not os.path.exists(prepross_path + 'validation/'):
	os.mkdir(prepross_path + 'validation/')

train = os.listdir(train_path)
val= os.listdir(val_path)
for folder in train:
	if folder[0] =='.':
		continue
	print train_path + folder
	if not os.path.exists(prepross_path + 'train/' + folder ):
	os.mkdir(prepross_path + 'train/' + folder)
	files = os.listdir(train_path + folder)
	# train_path =  train_path + folder
	for file in files:
		if file[0] =='.':
			continue
		filename = train_path + folder + '/' + file
		print filename
		

