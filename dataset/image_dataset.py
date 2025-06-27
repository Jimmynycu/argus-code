from torch.utils.data import Dataset
import os
import random
import torch
import numpy as np
from PIL import Image
import glob
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, rotation=False, height=512, width=1024, clip_info_path=None):
        
        IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.JPEG', '.JPG', '.PNG']

        self.images = []

        if clip_info_path is None:
            if type(root_dir) == str:
                root_dir = [root_dir]
            for root in root_dir:
                self.images.extend([x for x in glob.glob(root + '/**/*', recursive=True) if os.path.splitext(x)[1] in IMG_EXTENSIONS])
        else:
            if type(root_dir) == list:
                assert len(root_dir) == 1, "clip info path is only supported for a single folder"
                root_dir = root_dir[0]
            with open(clip_info_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    category, video_id, clip, img_name = line.strip().split('\t')
                    self.images.append(os.path.join(root_dir, category, video_id, clip, img_name+'.png'))
                    # self.images.extend([x for x in glob.glob(os.path.join(root_dir, category, video_id, clip, '*')) if os.path.splitext(x)[1] in IMG_EXTENSIONS])

        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor()
        ])
        self.rotation = rotation
        self.root_dir = root_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, height, width).
        """
        # Randomly select a folder (representing a video) from the base folder
        image_path = self.images[idx]
        while not os.path.exists(image_path):
            image_path = random.choice(self.images)
        image = Image.open(image_path)
        image = self.transform(image)

        if self.rotation: # roll the image along the width axis, as if it were a 360 image
            image = torch.roll(image, random.randint(0, image.shape[2]), dims=2)
            
        image = image * 2 - 1 # to range [-1, 1]

        ret = {'pixel_values': image, 'path': image_path}
                
        return ret