import cv2
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np

# DSM sub_3.tif
profile_list = [
    [(182,588), (182,675)],
    [(326,675), (410,675)],
    [(368,1586), (368,1670)],
    [(196,1589), (196,1668)],
]

dsm = rio.open('./sample/DSM/sub_3.tif').read(1)
# pan = rio.open('./sample/PAN/sub_3.tif').read(1)
label = rio.open('./sample/LABEL/sub_3.tif').read(1)
output = rio.open('./sample/OUTPUT/output-854_sub_3.tif').read(1)

'''
# draw profile line
dsm = cv2.cvtColor(dsm, cv2.COLOR_GRAY2BGR)
label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

for p in profile_list:
    dsm = cv2.line(dsm, p[0], p[1], (255,0,0),8)
    label = cv2.line(label, p[0], p[1], (255,0,0),8)
    output = cv2.line(output, p[0], p[1], (255,0,0),8)

for p in profile_list:
    # DSM
    plt.figure()
    plt.imshow(dsm, cmap='pink')
    plt.axis('off')
    plt.savefig('./sample/profile/DSM_sub_3.png')

    # LABEL
    plt.figure()
    plt.imshow(label, cmap='pink')
    plt.axis('off')
    plt.savefig('./sample/profile/LABEL_sub_3.png')

    # OUTPUT
    plt.figure()
    plt.imshow(output, cmap='pink')
    plt.axis('off')
    plt.savefig('./sample/profile/OUTPUT_sub_3.png')
'''

# Get value
for i, p in enumerate(profile_list):
    bg = np.zeros((dsm.shape), np.uint8)
    bg = cv2.line(bg, p[0], p[1], 255, 1)
    coords = np.argwhere(bg)
    
    d_values, l_values, o_values = [], [], []
    for c in coords:
        pxD = dsm[c[0], c[1]]
        pxL = label[c[0], c[1]]
        pxO = output[c[0], c[1]]

        d_values.append(pxD)
        l_values.append(pxL)
        o_values.append(pxO)

    plt.figure()
    plt.plot(d_values, c='blue', label='DSM')
    plt.plot(l_values, c='green', label='LABEL')
    plt.plot(o_values, c='red', label='OUTPUT')
    title = f'profile{i+1}'
    plt.title(title)
    plt.legend()
    plt.savefig(f'./sample/profile/{title}.png')






















    
