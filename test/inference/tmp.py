import matplotlib.pyplot as plt
import os
from glob import glob
import rasterio

dsm_list = sorted([x for x in glob('./sample/DSM/*.tif')])
pan_list = sorted([x for x in glob('./sample/PAN/*.tif')])
label_list = sorted([x for x in glob('./sample/LABEL/*.tif')])
output_list = sorted([x for x in glob('./sample/OUTPUT/*.tif')])


for d, p, l, o in zip(dsm_list, pan_list, label_list, output_list):
    tifs = [d, p, l, o]
    titles = ['DSM', 'PAN', 'LABEL', 'OUTPUT']
    fig = plt.figure(figsize=(20,20))
    for i, (tif, title) in enumerate(zip(tifs, titles)):
        rst = rasterio.open(tif)
        ax = fig.add_subplot(2,2, i+1)
        ax.set_title(title, fontsize=18)
        ax.imshow(rst.read(1), cmap='pink')
        ax.axis('off')
        rst.close()
    fig.tight_layout()
    fig.savefig(o.replace('.tif', '-TOTAL.png'))
    plt.close()

