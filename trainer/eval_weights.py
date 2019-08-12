import tensorflow as tf
import numpy as np
import tensorflow.contrib.opt as opt
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import matplotlib as mpl
mpl.use('TkAgg')
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

with open('/Users/xuerongwan/Documents/keras_job/weights/3hrs_weights', 'rb') as fp:
    weights_0 = pickle.load(fp)


with open('/Users/xuerongwan/Documents/keras_job/weights/6hrs_weights', 'rb') as fp:
    weights_1 = pickle.load(fp)

fig = plt.figure()
# for i in range(3):
#     ax = fig.add_subplot(2, 3, (i+1))
#     im = ax.imshow(weights_0[i*2], cmap='hot')
#     plt.colorbar(im, orientation='horizontal')
#
#     ax = fig.add_subplot(2, 3, (i+4))
#     im = ax.imshow(weights_1[i*2], cmap='hot')
#     plt.colorbar(im, orientation='horizontal')
#
#     ax = fig.add_subplot(3, 3, (i+7))
#     im = ax.imshow(weights_1[i*2]-weights_0[i*2], cmap='hot')
#     plt.colorbar(im, orientation='horizontal')

# plt.show()
i=1
ax = fig.add_subplot(1, 3, (i ))
im = ax.imshow(weights_0[i * 2], cmap='hot')
plt.colorbar(im, orientation='horizontal')

ax = fig.add_subplot(1, 3, (i + 1))
im = ax.imshow(weights_1[i * 2], cmap='hot')
plt.colorbar(im, orientation='horizontal')

ax = fig.add_subplot(1, 3, (i + 2))
im = ax.imshow(weights_1[i * 2][100:,0:100] - weights_0[i * 2], cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()