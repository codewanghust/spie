import numpy as np


a = '/Users/celine/CLL/pred_scale2.npy'



b = np.load(a)


t = []
tt = []


print (b)
print (b.shape)
print (b.shape[0])

c =  b[b[:,2].argsort()]
# for i in range(b.shape[0]):

print (c[18:])
print (c[18:].mean(axis=0))