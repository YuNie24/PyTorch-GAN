import torch.utils.data as data
import os
import torch

class Bedroom(data.Dataset):
    classes = ['0 - bedroom']

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))


    def scanImageFiles(self, images_path):
        imageFiles = []
        for root, dirs, files in os.walk(images_path):
            for file in files:
                if file.endswith(('.jpg', '.JPG', '.png', '.PNG')):
                    imageFiles.append(os.path.join(root, file))
        imageFiles.sort()
        return imageFiles

