import numpy as np
import os
import csv
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from Gabor import *
from tqdm import tqdm
import preprocess
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA


class AffectNet(object):
    def __init__(self):
        self.train_folder = '/kaggle/input/affectnet/train_set/train_set'
        self.val_folder = '/kaggle/input/affectnet/val_set/val_set'
        # self.csv_file = csv_file
        self.expressions = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt',
                            'None', 'Uncertain', 'No-face']
        self.filter = Gabor().build_filters()

    def get_imlist(self, csv_file):
        images = []
        names = []
        labels = []

        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                name = row['name']
                names.append(name)

                x_list = row['image'][1:-1].split()
                flattened_image = [int(x) for x in x_list]
                # img_shape = (int(np.sqrt(len(flattened_image))),) * 2
                img = np.array(flattened_image).reshape(224, 224)

                images.append(img)

                label = row['label']
                labels.append(label)

        return images, names, labels

    def gen_train_no(self):
        images = []
        nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = []
        lnds = []

        annotation_directory = os.path.join(self.train_folder, 'annotations')
        image_directory = os.path.join(self.train_folder, 'images')
        image_list = os.listdir(image_directory)
        for i in tqdm(range(len(image_list))):
            if image_list[i].endswith('.jpg') or image_list[i].endswith('.png') or image_list[i].endswith('.jpeg'):
                img_path = os.path.join(image_directory, image_list[i])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = preprocess.gray_norm(img)
                img = preprocess.adaptive_histogram_equalization(img)
                res = Gabor().getGabor(img, self.filter, False, 1)
                res = np.array(res).reshape(-1)
                index = image_list[i].split('.')[0]
                exp_file = os.path.join(annotation_directory, index + '_exp.npy')
                exp = np.load(exp_file).astype('int')
                lnd_file = os.path.join(annotation_directory, index + '_lnd.npy')
                lnd = np.load(lnd_file).astype('int').reshape(-1, 2)
                nums[exp] += 1
                images.append(res)
                labels.append(exp)
                lnds.append(lnd)
        x_train = np.array(images)
        nums = np.array(nums).astype('int')
        lnds = np.array(lnds)
        y_train = np.array(labels).astype('int')
        return self.expressions, x_train, y_train, nums, lnds

    def gen_valid_no(self):
        images = []
        nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        labels = []
        lnds = []

        annotation_directory = os.path.join(self.val_folder, 'annotations')
        image_directory = os.path.join(self.val_folder, 'images')
        image_list = os.listdir(image_directory)
        for i in tqdm(range(len(image_list))):
            if image_list[i].endswith('.jpg') or image_list[i].endswith('.png') or image_list[i].endswith('.jpeg'):
                img_path = os.path.join(image_directory, image_list[i])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = preprocess.gray_norm(img)
                img = preprocess.adaptive_histogram_equalization(img)
                res = Gabor().getGabor(img, self.filter, False, 1)
                res = np.array(res).reshape(-1)
                index = image_list[i].split('.')[0]
                exp_file = os.path.join(annotation_directory, index + '_exp.npy')
                exp = np.load(exp_file).astype('int')
                lnd_file = os.path.join(annotation_directory, index + '_lnd.npy')
                lnd = np.load(lnd_file).astype('int').reshape(-1, 2)
                nums[exp] += 1
                images.append(res)
                labels.append(exp)
                lnds.append(lnd)
        x_train = np.array(images)
        nums = np.array(nums).astype('int')
        lnds = np.array(lnds)
        y_train = np.array(labels).astype('int')
        return self.expressions, x_train, y_train, nums, lnds


class Gaborpipeline(BaseEstimator, TransformerMixin):
    def __init__(self, reduction=1):
        # self.data = dataset
        self.reduction = reduction

    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        train = []
        filters = Gabor().build_filters()
        for i in tqdm(np.arange(0, x.shape[0], 1)):
            x[i] = preprocess.gray_norm(x[i])
            x[i] = preprocess.adaptive_histogram_equalization(x[i])
            res = Gabor().getGabor(x[i], filters, False, self.reduction)
            res = np.array(res).reshape(-1)
            # res = np.append(res, y[i])
            train.append(res)
        train = np.array(train)
        # scaler = MinMaxScaler()
        # scaler.fit(train)
        # x_train = scaler.transform(train)
        return train, y


if __name__ == '__main__':
    dataset = AffectNet()
    # _, x_train, y_train = dataset.gen_train_no()
    _, x_val, y_val, nums_val = dataset.gen_valid_no()
    print(nums_val)

    num_pipeline = Pipeline([
        ('gabor', Gaborpipeline()),
        ('scaler', MinMaxScaler()),
        ('PCA', PCA(n_components=min(1000, min(x_val.shape[1] - 7, x_val.shape[0])))),
        ('svm', SVC())
    ])

    num_pipeline.fit(x_val, y_val)
    print(num_pipeline.score(x_val, y_val))
