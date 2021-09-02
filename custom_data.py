import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
# from utils import transform
import xml.etree.ElementTree as ET
import numpy as np

def toTensor(data):
    data = data.permute(2,0,1)
    return data

class CustomData(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self,  df, label_map, transform=None, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.df = df


        self.transform = transform
        # self.transform = target_transform

        self.keep_difficult = keep_difficult

        self.class_dict = label_map

    def __getitem__(self, index):
        # Read image

        image = self.df["images_path"].loc[index]
        ann = self.df["anns_path"].loc[index]

        image = Image.open(image, mode='r')
        image = image.convert('RGB')
        image = np.asarray(image)
        

        boxes, categories, is_difficult = self._get_annotation(ann)

        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            category = category[is_difficult == 0]

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=categories)
            image = transformed['image']
            boxes = transformed['bboxes']

        w, h = image.shape[:2]
        label = []
        for (x1,y1,x2,y2), category in zip(boxes, categories):
            x1 = x1/w
            y1 = y1/h 
            x2 = x2/w
            y2 = y2/h
            label.append([category, x1,y1,x2,y2])

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.tensor(label, dtype=torch.float32)


        return toTensor(image) , label
    
    def _get_annotation(self, ann):

        objects = ET.parse(ann).findall("object")

        boxes = []
        categories = []
        is_difficult = []
        for object in objects:

            class_name = object.find('name').text.lower().strip()

            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                categories.append(self.class_dict[class_name])

                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
        
        assert len(boxes) == len(categories)

        return boxes, categories, is_difficult


        # return (np.array(boxes, dtype=np.float32),
        #         np.array(category, dtype=np.float32),
        #         np.array(is_difficult, dtype=np.uint8))

    def __len__(self):
        return len(self.df)

    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        # difficulties = list()

        for b in batch:
            images.append(b[0])
            labels.append(b[1])
            # difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        
        return images, labels  # tensor (N, 3, 300, 300), 2 lists of N tensors each
