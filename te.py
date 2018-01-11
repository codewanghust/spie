import numpy as np


import argparse
import os
import numpy as np


def parse_inputs():
    parser = argparse.ArgumentParser(description='Test different nets with 3D data.')
  
    parser.add_argument('-mn', '--model_name', dest='model_name', type=str, default='dense24')
    parser.add_argument('-r', '--root', dest='root', type=str, default='')
    return vars(parser.parse_args())


options = parse_inputs()
model_name = options['model_name']
root = options['root']


dice_whole = np.load( root +'/' +   model_name + '_dice_whole.npy')[:,1]
dice_core = np.load( root + '/' + model_name + '_dice_core.npy')[:,1]
dice_et = np.load(root + '/' +  model_name + '_dice_enhance.npy')[:,1]

print dice_whole
print dice_whole.shape 

b = np.concatenate((dice_whole, dice_core,dice_et), axis=0)

print b.shape




# b = np.load(a)


# t = []
# tt = []


# print (b)
# print (b.shape)
# print (b.shape[0])

# c =  b[b[:,2].argsort()]
# # for i in range(b.shape[0]):

# print (c[18:])
# print (c[18:].mean(axis=0))