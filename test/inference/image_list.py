import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
from glob import glob
import rasterio
from rasterio.enums import Resampling

import osr
import ogr

import gdal
from osgeo import gdal_array

from sklearn.preprocessing import MinMaxScaler

from src.models import *
from src.networks import *
from src.Data import Data
from src.utils import *

import matplotlib.pyplot as plt
from time import time

# Set Tensorflow GPU Option
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options,
    allow_soft_placement=True,
    log_device_placement=True
    ))

parser = argparse.ArgumentParser(description='arguments input')
parser.add_argument('-d','--data', type=str, default='./sample/')
parser.add_argument('-w','--weights', type=str,default='../results/gen/weights-3951.hdf5',
    help='Path to HDF5 weights file')
parser.add_argument('-r', '--resize', type=str, default='DSM',
    help='DSM(default) or PAN: Resize to DSM or PAN')
args = parser.parse_args()

# padding of 16px here to avoid artefacts
pad = (64,64)
shape = (128,128) # 256 - 2 * 16
img_height, img_width = 256,256


# Load model
myModel = Wnet_cgan(img_height, img_width, n_labels=1)
myModel.build_wnet_cgan([64,64,128,128],
                    (3,3), 
                    wnet_activation='selu',
                    wnet_lr=1e-4,
                    discr_inp_channels = 16,
                    discr_block_list=[32,32,64,64],
                    discr_k_size=(3,3), 
                    discr_activation='relu',
                    discr_lr=1e-4,
                    lambda_=1e-1)

load_weights_time = time()
myModel.wnet_cgan.load_weights(args.weights)

# Set path
epoch = os.path.split(args.weights)[1].split('.')[0].split('-')[1]
output_folder = os.path.join(args.data, 'OUTPUT')
output_folder = os.path.join(output_folder, epoch)
# Make folder
os.makedirs(output_folder, exist_ok=True)

inp_dict = {}
for folder in [x for x in os.listdir(args.data) if x in ["DSM", "PAN", "LABEL"]]:
    folder_path = os.path.join(args.data, folder)
    inp_dict[folder] = sorted([x for x in glob(folder_path + "/*.tif")])

for dsm_path, pan_path, label_path in zip(inp_dict["DSM"], inp_dict["PAN"], inp_dict["LABEL"]):
    out_name = f"output_{os.path.split(dsm_path)[1]}"
    out_path = os.path.join(output_folder, out_name)


# Processing input
    dsm = rasterio.open(dsm_path).read()
    pan = rasterio.open(pan_path).read()
    ary_dsm = np.moveaxis(dsm, 0, -1)
    ary_pan = np.moveaxis(pan, 0, -1)

# Padding via mirroring to avoid artefacts
    padH = shape[0]*((ary_dsm.shape[0]//shape[0])+min(1,(ary_dsm.shape[0]%shape[0])))
    padW = shape[1]*((ary_dsm.shape[1]//shape[1])+min(1,(ary_dsm.shape[1]%shape[1])))

    h_pad = (pad[0],padH - ary_dsm.shape[0] + pad[0])
    w_pad = (pad[1],padW - ary_dsm.shape[1] + pad[1])

    padded_ary_dsm = np.pad(ary_dsm, pad_width=(h_pad, w_pad, (0, 0)), mode='symmetric')
    padded_ary_pan = np.pad(ary_pan, pad_width=(h_pad, w_pad, (0, 0)), mode='symmetric')

    origin2tile = time()
    W_dsm, _ = ary_to_tiles_forward(padded_ary_dsm, ary_dsm, pad=pad, shape=shape, scale=False)
    W_pan, _ = ary_to_tiles_forward(padded_ary_pan, ary_pan, pad=pad, shape=shape, scale=False)
    
    inf_time = time()
    W_hat_test = myModel.wnet.predict([W_dsm, W_pan], verbose=1)
    print('\tinference time ... ', time() - inf_time)


    # processing into tif file
    tile_to_tif = time()
    W_hat_ary = tiles_to_ary_forward(stacked_ary=W_hat_test, pad=pad, Gary_shape_padded=padded_ary_dsm.shape[:2])
    print('\ttile to tif time ... ', time() - tile_to_tif)

    post_proc_time = time()
    if args.resize == 'DSM':
        meta = rasterio.open(dsm_path).meta.copy()
    else:
        meta = rasterio.open(pan_path).meta.copy()

    meta.update({'count': 1, 'dtype':'float32'})

    '''
# Min/Max Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    ascolumns = W_hat_ary.reshape(-1, 1)
    t = scaler.fit_transform(ascolumns)
    pp_ = t.reshape(W_hat_ary.shape)
    '''
    # No Scaling
    pp_ = W_hat_ary
    with rasterio.open(out_path, 'w', **meta) as rr:
        rr.write(pp_[:pad[0]-h_pad[1], :pad[1]-w_pad[1],0].astype(np.float32),1)
    print('\tpost processing time ... ', time() - post_proc_time)

# Visualization
    draw_time = time()
    plt.figure()
    rst = rasterio.open(out_path).read(1)
    plt.imshow(rst, cmap='pink')
    plt.axis('off')
    plt.savefig(out_path.replace('tif', 'png'))
    plt.close()

    tifs = [dsm_path, pan_path, label_path, out_path]
    titles = ['DSM', 'PAN', 'LABEL', 'OUTPUT']
    fig = plt.figure(figsize=(20,20))
    for i, (tif, title) in enumerate(zip(tifs, titles)):
        rst = rasterio.open(tif)
        ax = fig.add_subplot(2,2, i+1)
        ax.set_title(title, fontsize=22)
        ax.imshow(rst.read(1), cmap='pink')
        ax.axis('off')
        rst.close()
    fig.tight_layout()
    fig.savefig(out_path.replace('.tif', '.png').replace('output', 'total'))
    plt.close()
    print('\tdraw time ... ', time()-draw_time)

