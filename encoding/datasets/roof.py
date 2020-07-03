###########################################################################
# Created by: Xiaoyi Liu
# Email: xiaoyi.liu@northwestern.edu
# Copyright (c) 2020
###########################################################################

import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import pickle

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from .base import BaseDataset

class RoofSegmentation(BaseDataset):
    BASE_DIR = 'RoofData'
    NUM_CLASS = 2
    def __init__(self, root=os.path.expanduser('~/.encoding/data'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(RoofSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists and prepare dataset automatically
        self.root = os.path.join(root, self.BASE_DIR)
        self.cache = os.path.join(root,self.BASE_DIR,'cache.pkl')

        assert os.path.exists(self.root), "Please setup the dataset"
        assert os.path.exists(self.cache), "Please setup the annotation cache using " + \
            "encoding/scripts/prepare_roof.py"
        
        with open(self.cache, 'rb') as f:
            cache = pickle.load(f)
        
        assert split in ['train','val'], "split can only be \'train\' or \'val\'"
        
        self.images, self.masks = cache['ext_id_list_'+split], cache['mask_list_'+split]
        
        if split != 'test':
            assert (len(self.images) == len(self.masks)), "number of images and masks does not match"
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        #if torch.is_tensor(index):
        #    index = index.tolist()
        img = Image.open(os.path.join(self.root,'images',self.images[index])).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # prepare masks
        polygons = self.masks[index]
        # init img_mask with '0'
        img_mask = Image.new('L', (img.width,img.height), 0)
        for polygon in polygons:
            poly_coords = [[i['x'], i['y']] for i in polygon]
            # add mask to img_mask
            polygon_tuple = [tuple(i) for i in poly_coords]
            ImageDraw.Draw(img_mask).polygon(polygon_tuple, outline=1, fill=1)
            mask = Image.fromarray(np.array(img_mask))  # [h,w]
        
        #sample = {'image': img, 'mask': mask}
        #if self.transform:
        #    sample = self.transform(sample)
        #return sample

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask


        


    #def _sync_transform(self, img, mask):
    #    # random mirror
    #    if random.random() < 0.5:
    #        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    #        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    #    crop_size = self.crop_size
    #    # random scale (short edge)
    #    w, h = img.size
    #    long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
    #    if h > w:
    #        oh = long_size
    #        ow = int(1.0 * w * long_size / h + 0.5)
    #        short_size = ow
    #    else:
    #        ow = long_size
    #        oh = int(1.0 * h * long_size / w + 0.5)
    #        short_size = oh
    #    img = img.resize((ow, oh), Image.BILINEAR)
    #    mask = mask.resize((ow, oh), Image.NEAREST)
    #    # pad crop
    #    if short_size < crop_size:
    #        padh = crop_size - oh if oh < crop_size else 0
    #        padw = crop_size - ow if ow < crop_size else 0
    #        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    #        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    #    # random crop crop_size
    #    w, h = img.size
    #    x1 = random.randint(0, w - crop_size)
    #    y1 = random.randint(0, h - crop_size)
    #    img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
    #    # gaussian blur as in PSP
    #    if random.random() < 0.5:
    #        img = img.filter(ImageFilter.GaussianBlur(
    #            radius=random.random()))
    #    # final transform
    #    return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') #- 1
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

'''
def _get_ade20k_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for filename in os.listdir(img_folder):
            basename, _ = os.path.splitext(filename)
            if filename.endswith(".jpg"):
                imgpath = os.path.join(img_folder, filename)
                maskname = basename + '.png'
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    print('cannot find the mask:', maskpath)
        return img_paths, mask_paths

    if split == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        print('len(img_paths):', len(img_paths))
        assert len(img_paths) == 20210
    elif split == 'val':
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        assert len(img_paths) == 2000
    else:
        assert split == 'trainval'
        train_img_folder = os.path.join(folder, 'images/training')
        train_mask_folder = os.path.join(folder, 'annotations/training')
        val_img_folder = os.path.join(folder, 'images/validation')
        val_mask_folder = os.path.join(folder, 'annotations/validation')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
        assert len(img_paths) == 22210
    return img_paths, mask_paths
'''