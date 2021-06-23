import h5py, os, glob
import numpy as np
from PIL import Image
from multiprocessing import Pool

DATAPATH = 'G:/data'
PATHOLOGYPATH = 'G:/data/pathology/'
MULTICENTER = ['foshan', 'shantou', 'guangzhou', 'huaxi']

def png_to_hdf5(sub):
    SRCTAPATH = f'{DATAPATH}/{center}/pathology'
    DSTPATH = f'{DATAPATH}/{center}/pathology_hdf5'
    sub_img_li = glob.glob(f'{SRCTAPATH}/{sub}/norm/*png')
    sub_img_li.sort()
    arr = []
    for sub_img in sub_img_li:
        arr.append(np.array(Image.open(sub_img)))
    arr = np.array(arr)
    os.makedirs(f"{DSTPATH}/{sub}", exist_ok=True)
    print(f'write:{sub}')
    with h5py.File(f"{DSTPATH}/{sub}/{sub}.hdf5", "w") as f:
        dataset = f.create_dataset("data", data=arr)
    del arr

if __name__ == '__main__':
    for center in MULTICENTER:
        sub_li = os.listdir(SRCTAPATH)

        pool = Pool(8)
        pool.map(png_to_hdf5, sub_li)
        pool.close()
        pool.join()
        print('Finish converting imgage')
