#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
@file: data_reader.py
@author: ImKe at 2022/7/18
@email: tuisaac163@gmail.com
@feature: #Enter features here
"""
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

class ImagefeatureCapDataset(Dataset):
    def __init__(self, caps:list, caps_fea: torch.Tensor, img_feas: torch.Tensor, cap_per_img:int =5):
        self.caps = caps
        self.caps_fea = caps_fea
        self.img_feas = img_feas
        self.cpi = cap_per_img
        self.initization(self.caps_fea, self.caps, self.cpi)

    def initization(self, caps_fea, caps, cap_per_img):
        self.uni_cap_fea = []
        self.uni_cap = []
        assert len(caps_fea) % cap_per_img == 0
        for i in range(int(len(caps_fea) / cap_per_img)):
            self.uni_cap_fea.append(caps_fea[i * cap_per_img])
            self.uni_cap.append(caps[i * cap_per_img])
        assert len(self.uni_cap_fea) == len(self.img_feas), f"There are {len(self.uni_cap)} captions, but {len(self.img_feas)} images"

    def __getitem__(self, index):
        cap_fea = self.uni_cap_fea[index]
        img_fea = self.img_feas[index]
        cap = self.uni_cap[index]
        return {'cap_fea': cap_fea, 'img_fea': img_fea, 'cap': cap}

    def __len__(self):
        return len(self.img_feas)


class DataListImageDataset(Dataset):
    def __init__(self, ids, load_path, captions, transform=transform, dataname="coco"):
        self.load_pth = load_path
        self.x = []
        self.y = []
        self.captions = captions
        self.t = transform
        self.initdata(ids, dataname)

    def initdata(self, ids, dataname):
        if dataname == "coco":
            for imgid in ids:
                img = '0' * (12 - len(imgid)) + imgid + ".jpg"
                self.x.append(os.path.join(self.load_pth, img))
                self.y.append(self.captions[str(imgid)])
        else:
            for imgid in ids:
                img = imgid
                self.x.append(os.path.join(self.load_pth, img))

    def __getitem__(self, index):
        x = self.x[index]
        x = self.t(Image.open(x).convert("RGB"))
        y = self.y[index]
        instance = {'img': x, 'captions': y}
        return instance

    def __len__(self):
        return len(self.x)

class DataListPreloadImageDataset(Dataset):
    def __init__(self, path_ids, captions, vis_path, dataname="coco"):
        """

        :param path_ids: str list
        :param captions: dict
        :param vis_path: pytorch file
        :param dataname: str
        """
        self.x = []
        self.y = []
        self.captions = captions
        self.vis_path = vis_path
        self.initdata(path_ids, dataname)

    def initdata(self, ids, dataname):
        img_feats = torch.load(self.vis_path)
        assert img_feats.shape[0]==len(ids)
        if dataname == "coco":
            for i, imgid in enumerate(ids):
                img = imgid.strip().split("/")[-1]
                self.x.append(img_feats[i])
                self.y.append(self.captions[str(int(img[:-4]))]) ## exclude .jpg
        else:
            for imgid in ids:
                img = imgid
                self.x.append(os.path.join(self.load_pth, img))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        instance = {'img': x, 'captions': y}
        return instance

    def __len__(self):
        return len(self.x)