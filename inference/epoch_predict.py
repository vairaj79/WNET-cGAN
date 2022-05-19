import os
from glob import glob


'''
Multi Epoch Test
'''

def epoch_list_test():
    # weights_path = '../results/gen/*.hdf5'
    # weights_list = [x for x in glob(weights_path)]

    weights_list = [f'../results/gen/weights-{str(x).zfill(2)}.hdf5' for x in [300]]
    d_path = '/home/ubuntu/project/seoul/dataset/sample_dataset/resize_dsm/DSM/sub_8.tif'
    p_path = d_path.replace('DSM', 'PAN')
    l_path = d_path.replace('DSM', 'LABEL')
    


    for w in weights_list:
        w_name = os.path.split(w)[1]
        print('='*len(w_name))
        print(w_name)
        print('='*len(w_name))
        cmd = f'python predict.py -w {w} -d {d_path} -p {p_path} -l {l_path}'
        os.system(cmd)

    print('Finished...')


'''
Multi Image Test
'''

def image_list_test():
    # Weights path
    W = '../results/gen/weights-2180.hdf5'
    # Get tif list
    dsm_list = sorted([x for x in glob('./sample/DSM/*.tif')])
    pan_list = sorted([x for x in glob('./sample/PAN/*.tif')])
    label_list = sorted([x for x in glob('./sample/LABEL/*.tif')])

    for D, P, L in zip(dsm_list, pan_list, label_list):
        cmd = f'python predict.py -d {D} -p {P} -l {L} -w {W}'
        os.system(cmd)

    print('Finished...')


if __name__ == '__main__':
    # epoch_list_test()
    image_list_test()


