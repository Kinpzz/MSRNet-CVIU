import os
import random
import numpy as np
from PIL import Image

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

class EdgeSalDataset(gluon.data.Dataset):

    def __init__(self,
                root,
                subset,
                input_size,
                transform,
                image_dir='imgs',
                label_dir='gt',
                image_set_dir='ImageSets',
                image_suffix='.jpg',
                label_suffix='.png',
                inference=False):
        self.input_size = input_size
        self.resize_bilinear = gluon.data.vision.transforms.Resize(size=(self.input_size, self.input_size), interpolation=1)
        self.resize_nearst = gluon.data.vision.transforms.Resize(size=(self.input_size, self.input_size), interpolation=0)
        self.transform = transform
        self.color_aug = transforms.RandomColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3)
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        self.inference = inference
        if not isinstance(root, list):
            root = [root]
            subset = [subset]
        label_name = label_dir
        self.images_path = []
        self.labels_path = []
        self.img_info = []
        for r, s in zip(root, subset):
            txt_fname = os.path.join(r, image_set_dir, s + '.txt')
            with open(txt_fname, 'r') as f:
                image_names = f.read().split()
            n = len(image_names)
            image_path, label_path, img_info = [None] * n, [None] * n, [None] * n
            for i in range(n):
                image_path[i] = os.path.join(r, image_dir, image_names[i] + image_suffix)
                label_path[i] = os.path.join(r, label_name, image_names[i] + label_suffix)
                if self.inference:
                    data = mx.image.imread(image_path[i])
                    img_info[i] = {'name': image_names[i], 'size': data.shape[0:2]}
            self.images_path += image_path
            self.labels_path += label_path
            self.img_info += img_info
            print(f"{root} {subset} has {n} images.")

    def __getitem__(self, idx):
        data = mx.image.imread(self.images_path[idx])
        if self.inference:
            data = self.resize_bilinear(data)
        else:
            label = mx.image.imread(self.labels_path[idx], flag=0)
            label = (label > 0).astype('float32')
            if self.transform:
                data = self.color_aug(data)
                data, label = rand_rotate(data, label, (-10, 10))
                data, label = rand_resized_crop(data, label, size=self.input_size, min_area=0.7, ratio=(0.7,1.3))
                data, label = rand_flip(data, label)
            else:
                data = self.resize_bilinear(data)
                new_label = mx.nd.zeros((self.input_size, self.input_size, label.shape[2]))
                for i in range(label.shape[2]):
                    new_label[:,:,i] = self.resize_nearst(label[:,:,i].expand_dims(axis=2))[:,:,0]
                label = new_label
                label = label.transpose((2,0,1))
        data = self.input_transform(data) # (w, h, c) -> (c, w, h)
        if self.inference:
            sample = data
        else:
            sample = data, label
        return sample

    def __len__(self):
        return len(self.images_path)

def rand_resized_crop(data, label, size, min_area, ratio):
    new_data, rect = mx.image.random_size_crop(data, (size,size), min_area, ratio)
    new_label = mx.nd.zeros((size, size, label.shape[2]))
    for i in range(label.shape[2]):
        new_label[:,:,i] = mx.image.fixed_crop(label[:,:,i].expand_dims(axis=2), rect[0], rect[1], rect[2], rect[3], (size, size), interp=0)[:,:,0]
    return new_data, new_label

def rand_flip(data, label):
    rand_flip_index = random.randint(0,2)
    if rand_flip_index < 2:
        data = mx.nd.flip(data, axis=rand_flip_index)
        for i in range(label.shape[2]):
            label[:,:,i] = mx.nd.flip(label[:,:,i], axis=rand_flip_index)
    return data, label

def rand_rotate(data, label, degree):
    deg = random.uniform(degree[0], degree[1])
    img = Image.fromarray(data.asnumpy())
    img = img.rotate(deg, resample=Image.BILINEAR)
    new_label = mx.nd.zeros(label.shape)
    for i in range(label.shape[2]):
        mask = Image.fromarray(label[:,:,i].asnumpy())
        new_label[:,:,i] = mx.nd.array(np.array(mask.rotate(deg, resample=Image.NEAREST)), mx.cpu(0))
    img = mx.nd.array(np.array(img), mx.cpu(0)).astype('uint8')
    return img, new_label
