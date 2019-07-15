# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import numpy as np
from imdb import Imdb
# import xml.etree.ElementTree as ET
import cv2
from PIL import Image
import numpy as np

class VocSegmentation(Imdb):
    """
    Implementation of Imdb for Pascal VOC segmentationd dataset
    Parameters:
    -----------
    image_set: str
        set to be used, can be train, testval, val
    year : str
        year of dataset, for Segmentation task, we choose 2012
    devkit_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load labels from SegmentationClass
    """
    def __init__(self, image_set, year, devkit_path, shuffle=False, is_train=True, names=''):
        super(VocSegmentation, self).__init__('VOCSegmentation' + '_' + image_set)
        self.image_set = image_set
        self.year = year
        self.devkit_path = devkit_path
        self.data_path = os.path.join(devkit_path, 'VOC' + year)
        self.raw_image_extension = '.jpg'
        self.label_image_extension = '.png'
        self.is_train = is_train
        self.classes = self._load_classes()
        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()


    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val or trainval)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, 'ImageSets', 'Segmentation', self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index

    def _label_path_from_index(self, index):
        """
        given image index, find out label path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, 'SegmentationClass', index + self.label_image_extension)
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x variable length] 2D tensor
        """
        temp = []

        # load labels from SegmentationClass
        num_loaded = 20 # for test
        for idx in self.image_set_index:
            label_file = self._label_path_from_index(idx)
            label = None
            label_img = Image.open(label_file)
            # convert [0,255] image to [0-1) np.array
            label_img = np.array(label_img).astype('float32') / 255.0
            assert label_img.ndim == 2, "label image should be 2-D tensor"
            label_img = np.swapaxes(label_img, 0, 1)
            for h_i in range(label_img.shape[0]):
                if label is None:
                    label = label_img[h_i,:]
                else:
                    # concat two arrays along axis=1 (column)
                    label = np.concatenate((label, label_img[h_i,:]))
            temp.append(label)

            num_loaded -= 1
            if num_loaded == 0:
                break
        self.num_images = 20
        return temp

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index]

    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, 'JPEGImages', name + self.raw_image_extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def _load_classes(self):
        return ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv']

