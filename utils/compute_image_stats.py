import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


class ImageDataset:
    def __init__(self, data_dir):
        self.images = self.get_all_images(data_dir)

    def get_all_images(self, folder):
        if os.path.isfile(folder):
            return [folder] if folder.endswith('.jpg') else []

        # Else must be directory
        files = []
        for f in os.listdir(folder):
            files += self.get_all_images(os.path.join(folder,f))
        return files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))/255
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image)
        return image


def main(data_dir):
    # Data loader
    df = ImageDataset(data_dir)
    image_loader = DataLoader(df,
                              batch_size=1,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)

    # Compute mean, std
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    count   = 0
    for inputs in tqdm(image_loader):
        count   += inputs.shape[0]*inputs.shape[2]*inputs.shape[3]
        psum    += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    #count = len(df) * image_size * image_size
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # Print
    print('count: '  + str(count))
    print('mean:  '  + str(total_mean))
    print('std:   '  + str(total_std))



if __name__=='__main__':
    main(sys.argv[1])
