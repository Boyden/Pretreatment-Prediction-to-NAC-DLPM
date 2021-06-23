import os, glob, sys, shutil, math, openslide, cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool

DATAPATH = 'G:/data'
PATHOLOGYPATH = 'G:/data/pathology/'
MULTICENTER = ['foshan', 'shantou', 'guangzhou', 'huaxi']

# tile image
def tile_img_png(img_path, overwrite=False):
    sub_id = os.path.basename(img_path).split('.')[0]
    level = 0
    downsample_level = 2
    dstdir = f"{PATHOLOGYPATH}/{center}/{sub_id}/tile"
    os.makedirs(dstdir, exist_ok=True)
    try:
        ptl_img = openslide.OpenSlide(img_path)
        downsample_ratio = ptl_img.level_downsamples[downsample_level]
        width, height = ptl_img.dimensions
        mpp = float(ptl_img.properties['openslide.mpp-x'])
        size = (int(512*0.25/mpp), int(512*0.25/mpp))

        x_tile_num = width//size[0]
        y_tile_num = height//size[1]

        img_downsample = ptl_img.get_thumbnail(ptl_img.level_dimensions[downsample_level])
        img = cv2.cvtColor(np.array(img_downsample), cv2.COLOR_RGB2HSV)
        img_s = img[:, :, 1]
        img_otsu, _ = cv2.threshold(img_s, 0, 255, cv2.THRESH_OTSU)
        img_s_mask = np.zeros(img_s.shape, dtype=np.uint8)
        img_s_mask[img_s>img_otsu] = 1
        kernel = np.ones((5,5),np.uint8)
        img_s_mask = cv2.morphologyEx(img_s_mask, cv2.MORPH_OPEN, kernel)

        x_tile_num, y_tile_num = int(x_tile_num), int(y_tile_num)
        for x in range(x_tile_num):
            for y in range(y_tile_num):
                img_name = f'{dstdir}/{sub_id}_{x}_{y}.png'

                if not os.path.exists(img_name) or overwrite is True:
                    x_min, y_min = int(x*size[0]), int(y*size[1])
                    x_min_mask, y_min_mask = int(x_min/downsample_ratio), int(y_min/downsample_ratio)
                    size_mask = (int(size[0]/downsample_ratio), int(size[1]/downsample_ratio))
                    patch_mask = img_s_mask[y_min_mask:y_min_mask+size_mask[1], x_min_mask:x_min_mask+size_mask[0]]
                    if patch_mask.sum() != 0: 
                        tmp_img = ptl_img.read_region((x_min, y_min), level, size).convert('RGB')
                        img_hsv = tmp_img.convert('HSV')
                        img_s = np.array(img_hsv)[:, :, 1]

                        # ratio: no color area ratio
                        if img_s.max() < 20:
                            img_s_hist = []
                            ratio = 1
                        else:
                            img_s_hist, _ = np.histogram(img_s, img_s.max())
                            ratio = sum(img_s_hist[:20])/sum(img_s_hist)

                        if ratio < 0.9:
                            print(f'tile image:{img_name}')
                            tmp_img_512 = tmp_img.resize((512, 512))
                            tmp_img_512.save(img_name)
                        else:
                            tmp_img_512 = None
                        del tmp_img, tmp_img_512, img_hsv, img_s, img_s_hist
    except Exception as e:
        print(f'error:{e}')
        print(f'error subject id:{sub_id}')

def remove_err_img(img_path):
    try:
        img = np.array(Image.open(img_path).convert("RGB"))
    except:
        print(f'remove image:{img_path}')
        os.remove(img_path)

if __name__ == '__main__':
    for center in MULTICENTER:
        print(f'Tile {center} WSIs')
        img_li = glob.glob(f'{PATHOLOGYPATH}/{center}/*/*svs') + glob.glob(f'{PATHOLOGYPATH}/{center}/*/*ndpi') + glob.glob(f'{PATHOLOGYPATH}/{center}/*/*tiff')
        pool = Pool(32)
        pool.map(tile_img_png, img_li)
        pool.close()
        pool.join()
        print('Finish tiling imgage')
