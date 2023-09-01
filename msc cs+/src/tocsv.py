import numpy as np
import os
import csv
import cv2
from tqdm import tqdm
import sys


expressions = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt',
                            'None', 'Uncertain', 'No-face']
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def write_csv_file(folder_path):
    image_folder = os.path.join(folder_path, 'images')
    annotation_folder = os.path.join(folder_path, 'annotations')
    imagelist = get_imlist(image_folder)
    x_train = []
    y_train = []
    for infile in imagelist:
        outfile = os.path.split(infile)[1]
        index = outfile.split('.')[0]
        im = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
        exp_file = os.path.join(annotation_folder, index + '_exp.npy')
        exp = np.load(exp_file)
        x_train.append(im)
        y_train.append(exp)
    x_train = np.array(x_train)
    y_train = np.array(y_train).astype('int')
    return x_train, y_train

def load_images_from_directory(directory):
    np.set_printoptions(threshold=sys.maxsize)
    images = []
    # labels = []
    indexes = []
    data_folder = "D:\\python_code\\dataset\\val_set"
    for i in range(len(expressions)):
        images.append(list())
        indexes.append(list())
        sub_folder = os.path.join(data_folder, expressions[i])
        if not os.path.exists(sub_folder):
            os.makedirs(os.path.join(data_folder, expressions[i]))
    annotation_directory = os.path.join(directory, 'annotations')
    image_directory = os.path.join(directory, 'images')
    image_list = os.listdir(image_directory)
    # for filename in os.listdir(image_directory):
    for i in tqdm(range(len(image_list))):
        if image_list[i].endswith('.jpg') or image_list[i].endswith('.png') or image_list[i].endswith('.jpeg'):
            img_path = os.path.join(image_directory, image_list[i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            index = image_list[i].split('.')[0]
            exp_file = os.path.join(annotation_directory, index + '_exp.npy')
            lnd_file = os.path.join(annotation_directory, index + '_lnd.npy')
            exp = np.load(exp_file).astype('int')
            lnd = np.load(lnd_file).astype('int').reshape(-1, 2)
            new_img = [img(x[0], x[1]) for x in lnd]
            # label = str(exp)
            # if img is not None:
            #     images[exp].append(img)
            #     indexes.append(index)
            # if labels is not None:
            #     labels.append(exp)
            folder = os.path.join(data_folder, expressions[exp])
            output_file = os.path.join(folder, index+'.csv')
            with open(output_file, 'w', newline='') as csvfile:
                fieldnames = ['name', 'image', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                flattened_img = img.flatten()
                writer.writerow({'name': index, 'image': flattened_img, 'label': exp})

    # images = np.array(images)
    # return images, labels, indexes
    # return images, indexes

def write_images_to_csv(images, labels, names, output_file):
    np.set_printoptions(threshold=sys.maxsize)
    folder = 'D:/python_code/msc cs+/dataset'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['name', 'image', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for name, img, label in zip(names, images, labels):
            flattened_img = img.flatten()
            writer.writerow({'name': name, 'image': flattened_img, 'label': label})


if __name__ == "__main__":
    # image_list = os.listdir('../dataset')
    aro = np.load('D:/tesr/train_set/annotations/100_lnd.npy')
    # maxs = np.clip(aro, 0, 22).astype('int')
    maxs = np.round(aro).astype('int')
    maxs = np.clip(maxs, 0, 223)
    maxs = maxs.reshape(-1, 2)
    print(maxs)
    img = cv2.imread('D:/tesr/train_set/images/100.jpg', cv2.IMREAD_GRAYSCALE)
    img = img[maxs[:, 0], maxs[:, 1]]
    print(img)
    # for x in maxs:
    #     # print(x)
    #     location = tuple(x)
    #     cv2.circle(img, location, 1, (255, 0, 0), 1)
    # cv2.imshow('name', img)
    # cv2.waitKey(0)  # 等待用户关闭图片窗口
    # cv2.destroyAllWindows()  # 关闭窗口
    # X_train, Y_train = write_csv_file('D:/tesr/train_set')
    # print(X_train.shape)
    # picture_directory = "D:/tesr/val_set/"
    # output_csv = 'trainset.csv'

    # load_images_from_directory(picture_directory)
    # write_images_to_csv(images_array, images_label, images_name, output_csv)
    # picture_directory = "D:/tesr/val_set/annotations"
    # print(cv2.cuda.getCudaEnabledDeviceCount())
    # img = cv2.imread('D:/tesr/val_set/images/1.jpg')
    # print(img.shape)
