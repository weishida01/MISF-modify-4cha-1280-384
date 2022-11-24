import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from .utils import create_mask
import random



class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist, mask_flist, training=True):
        super(Dataset, self).__init__()
        self.training = training
        self.data = self.load_flist(flist)
        self.edge_data = []
        self.mask_data = self.load_flist(mask_flist)

        self.sigma = 2
        self.mask = 3
        self.mask_threshold = 0

        print('training:{}  mask:{}  mask_list:{}  data_list:{}'.format(training, self.mask, mask_flist, flist))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
        img = Image.open(self.data[index])
        img = self.resize(img, 1280, 384)
        img_gray = img.convert('L')

        img = np.array(img)
        img_gray = np.array(img_gray)

        # load mask
        # mask = self.load_mask(index)
        mask = create_mask(img.shape[1], img.shape[0],
                           random.randint(40,80),
                           random.randint(40,80),
                           x=None, y=None)

        img = self.to_tensor(img)           # 0-255 变成了0-1
        img_gray = self.to_tensor(img_gray)             # 0-255 变成了0-1
        img_4 = torch.cat((img, img_gray), dim=0)

        mask = self.to_tensor(mask)      # 0-255 变成了0-1

        return img_4, mask

    def resize(self, img, height, width):
        imgh, imgw = img.size
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        box = (j, i, j + side, i + side)
        img = img.crop(box)
        img = img.resize((height, width))
        return img

    def load_mask(self,index):
        mask = np.array(Image.open(self.mask_data[index]))
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255       # threshold due to interpolation
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_flist(self, flist):
        if flist is None:
            return []
        with open(flist, 'r') as j:
            f_list = json.load(j)
            return f_list


    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item
