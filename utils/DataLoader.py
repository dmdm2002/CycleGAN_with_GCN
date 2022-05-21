import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random


def shuffle_folder(A_folders, B_folders):
    random.shuffle(B_folders)
    for i in range(len(A_folders)):
        if A_folders[i] == B_folders[i]:
            return shuffle_folder(A_folders, B_folders)
    return B_folders


class Loader(data.DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        super(Loader, self).__init__(self)
        self.dataset_dir = dataset_dir
        self.styles = styles
        A_folders = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')
        B_folders = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')

        B_folders = shuffle_folder(A_folders, B_folders)

        self.image_path_A = []
        self.image_path_B = []

        for folder in A_folders:
            temp = glob.glob(f'{folder}/*.png')
            self.image_path_A = self.image_path_A + temp

        for folder in B_folders:
            temp = glob.glob(f'{folder}/*.png')
            self.image_path_B = self.image_path_B + temp

        self.transform = transforms

    def __getitem__(self, index_A):
        index_B = random.randint(0, len(self.image_path_B)-1)

        item_A = self.transform(Image.open(self.image_path_A[index_A]))
        item_B = self.transform(Image.open(self.image_path_B[index_B]))

        return [item_A, item_B, self.image_path_A[index_A]]

    def __len__(self):
        return len(self.image_path_A)