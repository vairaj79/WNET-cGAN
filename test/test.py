import rasterio as rio
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import matplotlib.pyplot as plt


def min_max_scale(input_array):
    scaler = MinMaxScaler(feature_range=(0,1))
    ascolumns = input_array.reshape(-1, 1)
    t = scaler.fit_transform(ascolumns)
    result = t.reshape(input_array.shape)
    return result


def standardization(input_array):
    return (input_array - np.mean(input_array)) / np.std(input_array)

pan = rio.open('./origin_pan_sub_1.tif')

# arr_pan = pan.read(1)
arr_pan = pan.read(1, out_shape=(1,1500, 500))

# MinMaxScaling
mm_scaled = min_max_scale(arr_pan)
# mm_scaled = mm_scaled*255.

plt.figure()
plt.hist(mm_scaled.ravel(), 1000, [0, 1])
plt.savefig('minmax_hist.png')

# MaxScaling
max_scaled = arr_pan / np.max(arr_pan)
# max_scaled = max_scaled*255.

plt.figure()
plt.hist(max_scaled.ravel(), 1000, [0, 1])
plt.savefig('max_hist.png')

# meanstd scaling
std_scaled = standardization(arr_pan)
plt.figure()
plt.hist(std_scaled.ravel(), 1000, [np.min(std_scaled), np.max(std_scaled)])
plt.savefig('std_hist.png')

# plt.figure()
# plt.hist






'''
gray = pan.read(1)

plt.figure()
plt.hist(gray.ravel(), 100, [0, 1])
plt.show()
'''
