import torch
import numpy as np
import torch.utils.data as Data
import cv2
import os

def load_data_pair(data_path, case_name, modality='mr'):
    # Implement the function to load images and labels based on the case name and modality.
    image_path = os.path.join(data_path, f"{modality}_images", case_name)
    label_path = os.path.join(data_path, f"{modality}_labels", case_name)

    image_files = [f for f in os.listdir(image_path) if f.endswith('.jpg')]
    label_files = [f for f in os.listdir(label_path) if f.endswith('.jpg')]

    assert len(image_files) == len(label_files), "Number of images and labels do not match"

    # Load the images and labels in grayscale
    images = []
    labels = []
    for image_file, label_file in zip(image_files, label_files):
        image_full_path = os.path.join(image_path, image_file)
        label_full_path = os.path.join(label_path, label_file)

        image = cv2.imread(image_full_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_full_path, cv2.IMREAD_GRAYSCALE)
        label = np.where(label < 120, 0, 1)

        image = np.reshape(image, (1,) + image.shape)
        label = np.reshape(label, (1,) + label.shape)

        assert image is not None, f"Failed to load image: {image_full_path}"
        assert label is not None, f"Failed to load label: {label_full_path}"

        images.append(image)
        labels.append(label)

    return images, labels

def _to_tensor(data_dict):
    for name, data in data_dict.items():
        data_dict[name] = torch.from_numpy(data).float()
    return data_dict

class TrainDataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_path, evaluate=False, mode="MRtoUS"):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.img_keys = ['fixed_image', 'moving_image']
        self.dir = os.path.join(self.data_path,"mr_images")
        self.subject_list = sorted(os.listdir(self.dir))    #casexxxx
        self.subsubject_list = []
        for root, _, files in os.walk(self.dir):
            # 将每个文件的相对路径添加到列表中
            for name in files:
                relative_path = os.path.relpath(os.path.join(root, name), self.dir)
                self.subsubject_list.append(relative_path)

        # Get all cases from both modalities
        mr_cases = sorted(os.listdir(os.path.join(self.data_path, 'mr_images')))
        us_cases = sorted(os.listdir(os.path.join(self.data_path, 'us_images')))

        # Ensure the same cases exist in both modalities
        assert mr_cases == us_cases, "Cases do not match between modalities"

        # Pair images and labels with the same name across modalities
        self.filename = []
        for case in mr_cases:
            # Get image and label names for each case
            (mr_images, mr_labels) = load_data_pair(self.data_path, case, modality='mr')
            (us_images, us_labels) = load_data_pair(self.data_path, case, modality='us')

            # Ensure the same number of images and labels exists in both modalities
            assert len(mr_images) == len(us_images), "Different number of images for the case"
            assert len(mr_labels) == len(us_labels), "Different number of labels for the case"

            # Pair the images and labels
            if mode == "MRtoUS":
                paired_data = list(zip(us_images, mr_images, us_labels, mr_labels))
            elif mode == "UStoMR":
                paired_data = list(zip(mr_images, us_images, mr_labels, us_labels))
            else:
                raise TypeError("Mode has to be 'MRtoUS' or 'UStoMR'")
            self.filename.extend(paired_data)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        fix_img, mov_img, fix_label, mov_label = self.filename[index]

        # Create a dictionary to return

        data = {
            'fixed_image': fix_img,
            'moving_image': mov_img,
            'fixed_label': fix_label,
            'moving_label': mov_label,
        }

        return _to_tensor(data)