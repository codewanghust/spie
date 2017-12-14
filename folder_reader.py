import os
path = '/mnt/disk0/dat/lchen63/brain/newdata/'

train_path = path + 'Brats17TrainingData/'
#val_path = path + 'Brats17ValidationData/'

train_HGG_txt = path + 'train_HGG.txt'
validation_HGG_txt = path + 'val_HGG.txt'

train_HGG_writer = open(train_HGG_txt,'w')
val_HGG_writer = open(validation_HGG_txt,'w')


train_HGG = train_path  + 'HGG/'
train_LGG = train_path  + 'LGG/'

train_HGG_folder = os.listdir(train_HGG)
train_LGG_folder = os.listdir(train_LGG)
# validation_folder = os.listdir(val_path)
print len(train_HGG_folder)
print len(train_LGG_folder)
# print len(validation_folder)
counter = 0
for folder in train_HGG_folder:
	print train_HGG + folder
	if folder[0] == '.':
		continue

	counter += 1
	files = os.listdir(train_HGG + folder)
	for file in files:
		if file[0] == '.':
			coninue
		print  counter
		print train_HGG + folder + '/' + file
		if counter <=150:
			train_HGG_writer.write(train_HGG + folder + '/' + file + os.linesep)
		else:
			val_HGG_writer.write(train_HGG + folder + '/' + file + os.linesep)
print counter

# for folder in validation_folder:
# 	if folder[0] == '.':
# 		continue
# 	files = os.listdir(val_path + folder)
# 	for file in files:
# 		if file[0] == '.':
# 			continue
# 		print val_path + folder + '/' + file
# 	val_HGG_writer.write(val_path + folder + '/' + file + os.linesep)
