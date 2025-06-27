from torch.utils.data import Dataset
import os
import random
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor

class FrameDataset(Dataset):
    def __init__(self, base_folder, subdir_name=None,
                width=None, height=None, mast3r_img_size=None,
                sample_frames=25, frame_interval=3):
        """
        Args:
            channels (int): Number of channels, default is 3 for RGB.
            data format:
            self.base_folder
                ├── video_name1
                │   ├── video_frame1
                │   ├── video_frame2
                │   ...
                ├── video_name2
                │   ├── video_frame1
                    ├── ...
        """
        # Define the path to the folder containing video frames
        if type(base_folder) == list:
            assert len(base_folder) == 1
            base_folder = base_folder[0]
        self.base_folder = base_folder
        self.folders = [os.path.join(base_folder, folder) for folder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder))]
        if subdir_name is not None:
            self.folders = [os.path.join(x, subdir_name) for x in self.folders]

        self.width = width
        self.height = height
        self.mast3r_img_size = mast3r_img_size
        self.sample_frames = sample_frames
        self.frame_interval = frame_interval

        def check(folder):
            return len(os.listdir(folder)) >= (self.sample_frames - 1) * self.frame_interval + 1
        
        self.folders = list(filter(check, self.folders))

        print(f'Found {len(self.folders)} videos in {base_folder}')

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, height, width).
        """
        # Randomly select a folder (representing a video) from the base folder
        frame_names = os.listdir(self.folders[idx])
        frame_names.sort()

        def resize(frame: Image):
            if self.width is not None and self.height is not None:
                return frame.resize((self.width, self.height))
            if self.mast3r_img_size is not None: # return the longest side to be mast3r_img_size
                h, w = frame.size
                if h > w:
                    new_h = self.mast3r_img_size
                    new_w = int(w * self.mast3r_img_size / h)
                else:
                    new_w = self.mast3r_img_size
                    new_h = int(h * self.mast3r_img_size / w)
                return frame.resize((new_w, new_h))
            else:
                return frame

        frames = []

        # Load and process each frame
        for i, frame_name in enumerate(frame_names[::self.frame_interval]):
            frame_path = os.path.join(self.folders[idx], frame_name)
            img = Image.open(frame_path).convert('RGB')
            img = resize(img)
            frames.append(to_tensor(img) * 2 - 1)

            if i == self.sample_frames - 1:
                break
        
        pixel_values = torch.stack(frames)

        ret = {'frame_rate': 5, 'video': pixel_values, 'path': os.path.join(self.folders[idx], frame_names[0])}

        return ret