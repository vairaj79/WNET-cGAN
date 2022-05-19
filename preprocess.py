import rasterio as rio
import numpy as np

def preprocess(dsm_path, pan_path, label_path):
    dsm = rio.open(dsm_path)
    pan = rio.open(pan_path)
    label = rio.open(label_path)

    # Resize to DSM
    dsmW, dsmH = dsm.width, dsm.height

    # Meta info.
    Dmeta = dsm.meta.copy()
    Pmeta = pan.meta.copy()
    Lmeta = label.meta.copy()
    
    # Update PAN, LABEL meta info to DSM meta
    Dmeta.update({'dtype':'float32'})
    Pmeta.update({'width': dsmW, 'height': dsmH, 'count': 1, 'dtype':'float32'})
    Lmeta.update({'width': dsmW, 'height': dsmH, 'count': 1, 'dtype':'float32'})

    # Normalization
    ary_dsm = dsm.read()
    ary_pan = pan.read()
    ary_label = label.read()

    norm_dsm = ary_dsm / np.max(ary_dsm)
    norm_pan = ary_pan / np.max(ary_pan)
    norm_label = ary_label / np.max(ary_label)

    # Save result
    dsm_savename = dsm_path.replace('.tif', '_inp.tif')
    pan_savename = pan_path.replace('.tif', '_inp.tif')
    label_savename = label_path.replace('.tif', '_inp_tif')
    with rio.open(dsm_savename, 'w', **Dmeta) as dsm_data:
        dsm_data.write(norm_dsm.astype(np.float32))

    with rio.open(pan_savename, 'w', **Pmeta) as pan_data:
        pan_data.write(norm_pan.astype(np.float32))

    with rio.open(label_savename, 'w', **Lmeta) as label_data:
        label_data.write(norm_label.astype(np.float32))

if __name__ == '__main__':
    d = './inference/sample/testset/DSM_sub_2.tif'
    p = './inference/sample/testset/PAN_sub_2.tif'
    l = './inference/sample/testset/LABEL_sub_2.tif'
    preprocess(d, p, l)
