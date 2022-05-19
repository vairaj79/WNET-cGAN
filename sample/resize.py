import rasterio as rio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from glob import glob
def min_max_scale(input_array):
    scaler = MinMaxScaler(feature_range=(0,1))
    ascolumns = input_array.reshape(-1, 1)
    t = scaler.fit_transform(ascolumns)
    result = t.reshape(input_array.shape)
    return result


d_list = [x for x in glob('/home/ubuntu/project/seoul/dataset/sample_dataset/origin/DSM/*.tif')]
p_list = [x for x in glob('/home/ubuntu/project/seoul/dataset/sample_dataset/origin/PAN/*.tif')]
l_list = [x for x in glob('/home/ubuntu/project/seoul/dataset/sample_dataset/origin/LABEL/*.tif')]


for d, p, l in zip(d_list, p_list, l_list):
    dw = rio.open(d).read().shape[2]
    pw = rio.open(p).read().shape[2]
    lw = rio.open(l).read().shape[2]

    dl_ratio = lw / dw
    lp_ratio = pw / lw
    dp_ratio = pw / dw

    print(f'DSM-LABEL: {dl_ratio}\tLABEL-PAN: {lp_ratio}\tDSM-PAN: {dp_ratio}')
