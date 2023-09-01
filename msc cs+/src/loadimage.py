import numpy as np
import cv2 as cv
import dlib
from tqdm import tqdm
import os
import sys
import cv2
import pickle


class LandMark:
    def __init__(self, img):
        self.img = img

    def extract(self):
        detector = dlib.get_frontal_face_detector()
        dets = detector(self.img, 0)  # 使用detector进行人脸检测 dets为返回的结果

        predictor_path = "../model/shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
        shape_list = []
        landmark = []
        for index, face in enumerate(dets):
            shape = predictor(self.img, face)  # 寻找人脸的68个标定点
            # 遍历所有点，打印出其坐标，并用绿色的圈表示出来
            for _, pt in enumerate(shape.parts()):
                pt_pos = (pt.x, pt.y)
                landmark.append(pt_pos)
                cv.circle(self.img, pt_pos, 1, (0, 255, 0), 1)
            shape_list.append(shape)
        landmark = np.array(landmark)
        new_img = self.img[landmark[:, 0], landmark[:, 1]]
        new_img = np.array(new_img)
        return new_img


def load_images_from_directory(directory, number):
    images = []
    labels = []
    indexes = []
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    annotation_directory = os.path.join(directory, 'annotations' + str(number))
    image_directory = os.path.join(directory, 'images' + str(number))
    image_list = os.listdir(image_directory)
    for i in tqdm(range(len(image_list))):
        if image_list[i].endswith('.jpg') or image_list[i].endswith('.png') or image_list[i].endswith('.jpeg'):
            img_path = os.path.join(image_directory, image_list[i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            lnd = LandMark(img)
            index = image_list[i].split('.')[0]
            exp_file = os.path.join(annotation_directory, index + '_exp.npy')
            exp = np.load(exp_file).astype('int')
            features = lnd.extract()
            nums[exp] += 1
            index = int(index)
            images.append(features)
            labels.append(exp)
            indexes.append(index)
    images = np.array(images)
    labels = np.array(labels)
    indexes = np.array(indexes)

    return images, labels, indexes


if __name__ == '__main__':
    images, labels, names = load_images_from_directory('D:/python_code/dataset/subtrain_set', 0)
    with open(os.path.join('D:/python_code/msc cs+/dataset', 'subtrain0.pkl'), 'wb') as f:
        pickle.dump(images, f)
        pickle.dump(labels, f)

