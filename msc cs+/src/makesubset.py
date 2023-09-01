import os
import shutil
from tqdm import tqdm
import numpy as np

# Source and destination folder paths
source_folder = "D:\\tesr\\train_set"
destination_folder = "D:\\python_code\\dataset\\subtrain_set"

expressions = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt',
                            'None', 'Uncertain', 'No-face']
nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
baseline = [30350, 53766, 20368, 14000, 6300, 3800, 20368, 3700, 90, 90, 90]

batch_size = 3000
batch_index = 0
current_num = 0

image_source_folder = os.path.join(source_folder, 'images')
annotation_source_folder = os.path.join(source_folder, 'annotations')

image_destination_folder = os.path.join(destination_folder, 'images'+str(batch_index))
annotation_destination_folder = os.path.join(destination_folder, 'annotations'+str(batch_index))

os.makedirs(image_destination_folder, exist_ok=True)
os.makedirs(annotation_destination_folder, exist_ok=True)

# List all files in the source folder
file_list = os.listdir(image_source_folder)

# Loop through each file and transfer it to the destination folder
for i in tqdm(range(len(file_list))):
    source_image_path = os.path.join(image_source_folder, file_list[i])
    index = file_list[i].split('.')[0]
    source_exp_file = os.path.join(annotation_source_folder, index + '_exp.npy')
    source_lnd_file = os.path.join(annotation_source_folder, index + '_lnd.npy')
    exp = np.load(source_exp_file).astype('int')
    if nums[exp] < baseline[exp]:
        # Use shutil.copy to copy the file from source to destination
        destination_exp_file = os.path.join(annotation_destination_folder, index + '_exp.npy')
        destination_lnd_file = os.path.join(annotation_destination_folder, index + '_lnd.npy')
        destination_image_path = os.path.join(image_destination_folder, file_list[i])
        shutil.copy(source_image_path, destination_image_path)
        shutil.copy(source_exp_file, destination_exp_file)
        shutil.copy(source_lnd_file, destination_lnd_file)
        nums[exp] += 1
        current_num += 1
    if(current_num == batch_size):
        batch_index += 1
        image_destination_folder = os.path.join(destination_folder, 'images' + str(batch_index))
        annotation_destination_folder = os.path.join(destination_folder, 'annotations' + str(batch_index))
        os.makedirs(image_destination_folder, exist_ok=True)
        os.makedirs(annotation_destination_folder, exist_ok=True)
        current_num = 0

    # print(f"Transferred: {file_list[i]}")

print("Transfer completed.")
