
import os, glob, sys, shutil, math
import pandas as pd
import numpy as np
from pypinyin import pinyin, lazy_pinyin
from PIL import Image, ImageFilter, ImageChops


DATAPATH = 'G:/data'
MRIPATH = 'G:/data/mri/'
PATHOLOGYPATH = 'G:/data/pathology/'
MULTICENTER = ['foshan', 'shantou', 'guangzhou']


import openslide, cv2
import matplotlib.pyplot as plt
from lxml import etree
from PIL import Image, ImageFilter
from multiprocessing import Pool

# In[ ]:
# get pathological image with mask
def generate_ptl_img(sub_df):
    ptl_path = f"G:/data/ptl_img/guangzhou/total_ptl_img"
    ptl_id = sub_df['pathological_id']
    ptl_img = sub_df['pathological_img']
    pCR = int(sub_df['pCR'])
    if ';' in ptl_img:
        ptl_img = ptl_img.split(';')[0]
    ptl_xml = ptl_img.split('.')[0]+'.xml'
    try:
        ptl_img = openslide.OpenSlide(ptl_img)
        ptl_points = xml2points(ptl_xml)
        if not os.path.exists(f'{ptl_path}/{ptl_id}'):
            os.mkdir(f'{ptl_path}/{ptl_id}')
            os.mkdir(f'{ptl_path}/{ptl_id}/1')
            os.mkdir(f'{ptl_path}/{ptl_id}/0')
        for key in ptl_points.keys():
            for i, line in enumerate(ptl_points[key]):
                if not os.path.exists(f'{ptl_path}/{ptl_id}/{key}/{ptl_id}_pCR{pCR}_label_{key}_{i}.tif') or not os.path.exists(f'{ptl_path}/{ptl_id}/{key}/{ptl_id}_pCR{pCR}_label_{key}_{i}_mask.tif'):
                    x_min, x_max, y_min, y_max = min_rectangle_point(line, dim=2, padding=100)
                    tmp_img = ptl_img.read_region((x_min, y_min), 0, (x_max-x_min, y_max-y_min)).convert('RGB')
                    tmp_mask =  origin2patch_mask(line, (x_min, y_min), tmp_img.size)
                    tmp_img.save(f'{ptl_path}/{ptl_id}/{key}/{ptl_id}_pCR{pCR}_label_{key}_{i}.tif')
                    print(f'\nsave file:{ptl_path}/{ptl_id}/{key}/{ptl_id}_pCR{pCR}_label_{key}_{i}.tif\n')

                    tmp_mask_img = Image.fromarray(tmp_mask)
                    tmp_mask_img.save(f'{ptl_path}/{ptl_id}/{key}/{ptl_id}_pCR{pCR}_label_{key}_{i}_mask.tif')
                    del tmp_img
                    del tmp_mask
                    del tmp_mask_img
        del ptl_img
    except:
        print('err img:', ptl_img)

# tile image
def tile_img(img_path, threshold=0.0625, rewrite=False):
    size=512
    file_path = os.path.dirname(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    file_mask = f'{file_name}_mask.tif'
    Image.MAX_IMAGE_PIXELS = 10000000000
    try:
        img = Image.open(img_path).convert('RGB')
        img_mask = Image.open(f'{file_path}/{file_mask}')
        if not os.path.exists(f"{file_path}/tile"):
            os.mkdir(f"{file_path}/tile")
        x_dim, y_dim = img.size
        x_tile_num = math.ceil(x_dim/size)
        y_tile_num = math.ceil(y_dim/size)
        for x in range(x_tile_num):
            for y in range(y_tile_num):
                left, upper, right, lower = x*size, y*size, x*size+size, y*size+size
                tmp_img = img.crop((left, upper, right, lower))
                tmp_mask = img_mask.crop((left, upper, right, lower))
                content_area = np.array(tmp_mask).sum()
                if content_area/(size*size) >= threshold:
                    if right > x_dim or lower > y_dim:
                        x_shift = 0 if right < x_dim else (right - x_dim)//2
                        y_shift = 0 if lower < y_dim else (lower - y_dim)//2
                        tmp_img = ImageChops.offset(tmp_img, x_shift, y_shift)
                    
                    if not os.path.exists(f"{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.tif") or rewrite is True:
                        print(f"tile:{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.tif")
                        tmp_img.save(f"{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.tif")
                del tmp_img, tmp_mask
        del img, img_mask
    except:
        print(f'err img:{file_name}')

# tile image
def tile_img_png(img_path, rewrite=False):
    size=512
    file_path = os.path.dirname(img_path)
    file_name = os.path.basename(img_path).split('.')[0]
    Image.MAX_IMAGE_PIXELS = 10000000000
    try:
        img = Image.open(img_path).convert('RGB')
        if not os.path.exists(f"{file_path}/tile"):
            os.mkdir(f"{file_path}/tile")
        x_dim, y_dim = img.size
        # x_tile_num = math.ceil(x_dim/size)
        # y_tile_num = math.ceil(y_dim/size)
        x_tile_num = int(x_dim/size)
        y_tile_num = int(y_dim/size)
        for x in range(x_tile_num):
            for y in range(y_tile_num):
                if not os.path.exists(f"{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.png") or rewrite is True:
                    left, upper, right, lower = x*size, y*size, x*size+size, y*size+size
                    tmp_img = img.crop((left, upper, right, lower))
                    img_hsv = tmp_img.convert('HSV')
                    img_s = np.array(img_hsv)[:, :, 1]
                    if img_s.max() < 20:
                        img_s_hist = []
                        ratio = 1
                    else:
                        img_s_hist, _ = np.histogram(img_s, img_s.max())
                        ratio = sum(img_s_hist[:20])/sum(img_s_hist)
                    if ratio < 0.9:
                            print(f"tile:{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.png")
                            tmp_img.save(f"{file_path}/tile/{file_name}_{size}_tile_{x}_{y}.png")
                    del tmp_img, img_hsv, img_s, img_s_hist
        del img
    except:
        print(f'err img:{file_name}')

def tile_img_overlap(img_path, size=(512, 512), overlap=0.125, overwrite=False):
    sub_id = os.path.basename(img_path).split('.')[0]
    level = 0
    downsample_level = 2
    dstdir = f"/data_8t/bao/{center}/pathology_{str(overlap).split('.')[1]}/{sub_id}"
    os.makedirs(dstdir, exist_ok=True)
    try:
        ptl_img = openslide.OpenSlide(img_path)
        downsample_ratio = ptl_img.level_downsamples[downsample_level]
        width, height = ptl_img.dimensions
        # 512*0.75*x + 512*0.25 = width
        x_tile_num = (width-size[0]*overlap)/(size[0]*(1-overlap))
        y_tile_num = (height-size[1]*overlap)/(size[1]*(1-overlap))

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
                    x_min, y_min = int(x*size[0]*(1-overlap)), int(y*size[1]*(1-overlap))
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
                            tmp_img_128 = tmp_img.resize((128, 128))
                            tmp_img_128.save(img_name)
                        else:
                            tmp_img_128 = None
                        del tmp_img, tmp_img_128, img_hsv, img_s, img_s_hist
    except Exception as e:
        print(f'error:{e}')
        print(f'error subject id:{sub_id}')

def tile_img_overlap_tiff(img_path, size=(256, 256), overlap=0.125, overwrite=False):
    sub_id = os.path.basename(img_path).split('.')[0]
    level = 0
    dstdir = f"/data_8t/bao/{center}/pathology_{str(overlap).split('.')[1]}/{sub_id}"
    os.makedirs(dstdir, exist_ok=True)
    try:
        ptl_img = openslide.OpenSlide(img_path)
        width, height = ptl_img.dimensions
        # 512*0.75*x + 512*0.25 = width
        x_tile_num = (width-size[0]*overlap)/(size[0]*(1-overlap))
        y_tile_num = (height-size[1]*overlap)/(size[1]*(1-overlap))

        x_tile_num, y_tile_num = int(x_tile_num), int(y_tile_num)
        for x in range(x_tile_num):
            for y in range(y_tile_num):
                img_name = f'{dstdir}/{sub_id}_{x}_{y}.png'

                if not os.path.exists(img_name) or overwrite is True:
                    x_min, y_min = int(x*size[0]*(1-overlap)), int(y*size[1]*(1-overlap))

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
                        tmp_img_128 = tmp_img.resize((128, 128))
                        tmp_img_128.save(img_name)
                    else:
                        tmp_img_128 = None
                    del tmp_img, tmp_img_128, img_hsv, img_s, img_s_hist
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
    
    # pool = Pool(4)
    # pool.map(generate_ptl_img, sub_df_li)
    # pool.close()
    # pool.join()
    # for sub_df in sub_df_li:
    #     generate_ptl_img(sub_df)
    # print('Finish croping image')
    center = 'foshan'
    DATAPATH = f'/data_8t/bao/{center}'
    img_li = glob.glob(f'{DATAPATH}/*/*/*svs')
#    img_li = glob.glob(f'{DATAPATH}/*/*/tile/*png')
#    img_li = list(set(glob.glob(f'{DATAPATH}/*/*/*tif'))-set(glob.glob(f'{DATAPATH}/*/*/*mask.tif')))
    pool = Pool(32)
    pool.map(tile_img_overlap, img_li)
    pool.close()
    pool.join()
    print('Finish tiling imgage')
