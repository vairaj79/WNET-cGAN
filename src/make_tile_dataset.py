import os
from itertools import product
import rasterio as rio
from rasterio import windows
from rasterio.enums import Resampling
from tqdm import tqdm
from glob import glob
import shutil as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_tiles(ds, width=256, height=256):
    ncols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def division(input_fullname, output_path):
    # Set in/output path
    in_folder_path, in_file = os.path.split(input_fullname)
    tif_type = os.path.split(in_folder_path)[1]

    output_folder = os.path.join(output_path, tif_type)

    # Read origin tif file
    with rio.open(input_fullname) as inds:
        meta = inds.meta.copy()

        # Division
        for window, transform in get_tiles(inds, 256, 256):
            # Update tile metda info.
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            if (window.width == 256) and (window.height == 256):
                # Output file name
                out_file = '{}-{}_{}'.format(int(window.col_off),int(window.row_off),in_file)
                # Output full path
                output_fullname = os.path.join(output_folder, out_file)
                # Save tile tif file
                with rio.open(output_fullname, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))


def min_max_scale(input_array):
    scaler = MinMaxScaler(feature_range=(0,1))
    ascolumns = input_array.reshape(-1, 1)
    t = scaler.fit_transform(ascolumns)
    result = t.reshape(input_array.shape)
    return result

def max_norm(input_array):
    return input_array / np.max(input_array)


def main(base_path, resize_to, tile, multiple_tile):

    # Set data path
    # base_path = '/home/ubuntu/project/seoul/dataset/sample_dataset/origin'
    pan_path = os.path.join(base_path, 'PAN')
    dsm_path = os.path.join(base_path, 'DSM')
    label_path = os.path.join(base_path, 'LABEL')
    resize_path = os.path.join(os.path.split(base_path)[0], 'resize')
    tile_path = os.path.join(os.path.split(base_path)[0], 'tile')
    # Set folder
    os.makedirs(resize_path, exist_ok=True)
    os.makedirs(tile_path, exist_ok=True)
    for folder_name in ['PAN', 'DSM', 'LABEL']:
        os.makedirs(os.path.join(resize_path, folder_name), exist_ok=True)
        os.makedirs(os.path.join(tile_path, folder_name), exist_ok=True)

    # tif list
    pan_tif = sorted([x for x in glob(pan_path + '/*.tif')])
    dsm_tif = sorted([x for x in glob(dsm_path + '/*.tif')])
    label_tif = sorted([x for x in glob(label_path + '/*.tif')])
    
    resize_list = []
    # Resize : PAN(base), DSM, LABEL size match
    for p, d, l in zip(pan_tif, dsm_tif, label_tif):
        # print(p.split('/')[-1], d.split('/')[-1], l.split('/')[-1])

        # Read tif image
        pan = rio.open(p)
        dsm = rio.open(d)
        label = rio.open(l)

        # Base Shape
        if resize_to == 'DSM':
            ph, pw = dsm.meta['height'], dsm.meta['width']
        else:
            ph, pw = pan.meta['height'], pan.meta['width']

        # Resize to multiple tile size(256)
        if multiple_tile:
            if ph % tile != 0:
                ph = int(tile * ((ph // tile) + 1))
            if pw % tile != 0:
                pw = int(tile * ((pw // tile) + 1))

        # Get meta info
        p_meta = pan.meta.copy()
        d_meta = dsm.meta.copy()
        l_meta = label.meta.copy()

        # Update Meta info
        p_meta.update({'height': ph, 'width': pw, 'count':1, 'dtype':'float32'})
        d_meta.update({'height': ph, 'width': pw, 'count':1, 'dtype':'float32'})
        l_meta.update({'height': ph, 'width': pw, 'count':1, 'dtype':'float32'})

        # Set save path
        out_p = p.replace('origin', 'resize')
        out_d = d.replace('origin', 'resize')
        out_l = l.replace('origin', 'resize')

        # Normalization
        '''
        scaled_pan = min_max_scale(pan.read(out_shape=(1,ph,pw)))
        scaled_dsm = min_max_scale(dsm.read(out_shape=(1,ph,pw)))
        scaled_label = min_max_scale(label.read(out_shape=(1,ph,pw)))
        '''
        scaled_pan = max_norm(pan.read(out_shape=(1,ph,pw)))
        scaled_dsm = max_norm(dsm.read(out_shape=(1,ph,pw)))
        scaled_label = max_norm(label.read(out_shape=(1,ph,pw)))
        # Save resize image
        with rio.open(out_p, 'w', **p_meta) as p_data: # PAN
            p_data.write(scaled_pan.astype(np.float32))

        with rio.open(out_d, 'w', **d_meta) as d_data: # DSM
            d_data.write(scaled_dsm.astype(np.float32))

        with rio.open(out_l, 'w', **l_meta) as l_data: # LABEL
            l_data.write(scaled_label.astype(np.float32))
        # Total tif list
        resize_list.extend([out_p, out_d, out_l])

    print('Finished Resize image...')

    # Split resized image into tiles
    for tif in tqdm(resize_list, desc='to tile...'):
        division(tif, tile_path)


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datapath', type=str,
        default='/home/ubuntu/project/seoul/dataset/sample_dataset/origin',
        help='Upper path including DSM, PAN, LABEL folder.')

    parser.add_argument('-r','--resize', type=str, default='DSM',
        help='PAN or DSM(default). Resize to PAN or DSM size')

    parser.add_argument('-t','--tile', type=int, default=256,
        help='Tile size. 256(default)')

    parser.add_argument('-m','--multiple', type=bool, default=False,
        help='True or False(default). If True, resize to fit tile size.')

    opt = parser.parse_args()
    
    # Run
    main(opt.datapath, opt.resize, opt.tile, opt.multiple)



























