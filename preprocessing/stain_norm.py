import os, glob
import pandas as pd
import numpy as np
from PIL import Image
from skimage.color import rgb2hed
from multiprocessing import Pool
import staintools

MULTICENTER = ['foshan', 'shantou', 'guangzhou', 'huaxi']
PATHOLOGYPATH = 'G:/data/pathology/'
target_arr = np.load('stain_norm_target.npy')

def norm_img(img_path, overwrite=False):
    img_basename = os.path.basename(img_path)
    img_name = img_basename.split('.')[0]
    dst_name = f'{img_name}_norm.png'
    sub_id = img_basename.split('_')[0]
    dst_path = f'{PATHOLOGYPATH}/{center}/'
    dst_filename = f'{dst_path}/{sub_id}/norm/{dst_name}'
    if not os.path.exists(dst_filename) or overwrite is True:
        os.makedirs(f'{dst_path}/{sub_id}', exist_ok=True)

        img = np.array(Image.open(img_path))
        normalizer = staintools.StainNormalizer(method='vahadane')
        normalizer.fit(target_arr)
        norm_img = normalizer.transform(img)
        print(f'save img:{dst_name}')
        Image.fromarray(norm_img).save(dst_filename)

if __name__ == '__main__':
    for center in MULTICENTER:
        print(f'Norm {center} tiles')
        img_li = glob.glob(f'{PATHOLOGYPATH}/*/tile/*png')

        pool = Pool(32)
        pool.map(norm_img, img_li)
        pool.close()
        pool.join()
        print('Finish norm imgage')
