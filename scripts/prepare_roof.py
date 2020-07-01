''' Prepare Roof Data Set'''
import os
import shutil
import argparse
import zipfile
import pickle

import csv
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image, ImageDraw

_TARGET_DIR = os.path.expanduser('~/.encoding/data')
_RAND_SEED = 999

def copytree2(source,dest):
    os.mkdir(dest)
    dest_dir = os.path.join(dest,os.path.basename(source))
    shutil.copytree(source,dest_dir)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Roof dataset.',
        epilog='Example: python prepare_roof.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv', default=os.path.expanduser('~/RoofData/export-from-labelbox.csv'), help='path to csv annotation file')
    parser.add_argument('--img_dir', default=os.path.expanduser('~/RoofData/images'), help='path to images')
    args = parser.parse_args()
    return args

def export_mask_file(image_path, polygon, mask_file_path):
    pass

if __name__ == "__main__":
    args = parse_args()
    csv_path = args.csv
    image_root = args.img_dir

    mask_list = []
    ext_id_list = []
    info_list = []
    csv_dict = csv.DictReader(open(csv_path))
    for n, row in enumerate(csv_dict):
        if n>1:
            try:
                mask_list.append([obj['polygon'] for obj in json.loads(row['Label'])['objects']])
                ext_id_list.append(row['External ID'])
                info_list.append(row)
            except:
                pass
    num_images = len(mask_list)
    print("total number of images: {}".format(num_images))

    idx = np.arange(num_images)
    np.random.seed(_RAND_SEED)
    np.random.shuffle(idx)

    num_train = int(0.9 * num_images)

    mask_list_train = [mask_list[i] for i in idx[:num_train]]
    mask_list_val = [mask_list[i] for i in idx[num_train:]]

    ext_id_list_train = [ext_id_list[i] for i in idx[:num_train]]
    ext_id_list_val = [ext_id_list[i] for i in idx[num_train:]]

    data_to_pickle = {
        'mask_list_train': mask_list_train,
        'mask_list_val': mask_list_val,
        'ext_id_list_train': ext_id_list_train,
        'ext_id_list_val': ext_id_list_val,
    }

    dest_dir = os.path.join(_TARGET_DIR,'RoofData')
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    # copy all images
    copytree2(image_root, dest_dir)
    # cache split and annotations
    with open(os.path.join(dest_dir, 'cache.pkl'), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data_to_pickle, f, pickle.HIGHEST_PROTOCOL)

    # prepare mask files

    # 
