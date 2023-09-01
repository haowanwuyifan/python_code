import numpy as np
import cv2
import os
import cv2 as cv
from tqdm import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def load_images_from_directory(directory, num):
    #     np.set_printoptions(threshold=sys.maxsize)
    images = []
    labels = []
    indexes = []
    nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    cur = 0
    annotation_directory = os.path.join(directory, 'annotations')
    image_directory = os.path.join(directory, 'images')
    image_list = os.listdir(image_directory)
    for i in tqdm(range(len(image_list))):
        if image_list[i].endswith('.jpg') or image_list[i].endswith('.png') or image_list[i].endswith('.jpeg'):
            img_path = os.path.join(image_directory, image_list[i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            index = image_list[i].split('.')[0]
            exp_file = os.path.join(annotation_directory, index + '_exp.npy')
            exp = np.load(exp_file).astype('int')
            lnd_file = os.path.join(annotation_directory, index + '_lnd.npy')
            lnd = np.load(lnd_file)
            maxs = np.round(lnd).astype('int')
            maxs = np.clip(maxs, 0, 223)
            maxs = maxs.reshape(-1, 2)
            features = img[maxs[:, 0], maxs[:, 1]]
            nums[exp] += 1
            index = int(index)
            images.append(features)
            labels.append(exp)
            indexes.append(index)
            cur += 1
        if cur == num:
            break
    images = np.array(images)
    labels = np.array(labels)
    indexes = np.array(indexes)

    return images, labels, indexes


if __name__ == '__main__':
    expressions = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt',
                   'None', 'Uncertain', 'No-face']
    images, labels, indexes = load_images_from_directory('D:\\tesr\\val_set', 300)
    with open('D:\\python_code\\msc cs+\\model\\lndself.pkl', 'rb') as f:
        grid_search = pickle.load(f)
    predictions = grid_search.predict(images)

    # Find indices of wrong predictions
    wrong_indices = np.where(predictions != labels)[0]

    # Plot a few of the wrongly predicted images
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 8))  # Change nrows and ncols based on your preference

    for i, ax in enumerate(axes.ravel()):
        if i >= len(wrong_indices):
            break
        idx = wrong_indices[i]
        pic_name = os.path.join('D:\\tesr\\val_set\\images', str(indexes[idx])+'.jpg')
        pic = cv2.imread(pic_name, cv2.IMREAD_GRAYSCALE)
        ax.imshow(pic, cmap='gray')  # Adjust the shape based on your image dimensions
        ax.set_title(f"True: {expressions[labels[idx]]}, Pred: {expressions[predictions[idx]]}")
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()
