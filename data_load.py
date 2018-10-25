import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_table, root_dir, transform=None):
        """
        Args:
            data_table (pandas table): Data read from csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = data_table
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        central_crop (bool): Output a crop from the image center only
    """

    def __init__(self, output_size, central_crop=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.central_crop = central_crop

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        if self.central_crop:
            top = (h - new_h)//2
            left = (w - new_w)//2
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class RandomRotate(object):
    """Rotate randomly the image

    Args:
        max_ang (float): Max angle to rotate an image in deg
    """

    def __init__(self, max_ang=5):
        self.max_ang = max_ang

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        c_x, c_y = int(image.shape[0]/2), int(image.shape[1]/2)
        ang = self.max_ang*np.random.rand()-self.max_ang
        M = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
        image = cv2.warpAffine(image, M, image.shape[:2])
        
        key_pts = np.reshape(np.array(key_pts), (68, 1, 2))
        key_pts = cv2.transform(key_pts, M)
        key_pts = np.float32(np.reshape(key_pts, (68, 2)))

        return {'image': image, 'keypoints': key_pts}


class RandomBrCont(object):
    """Add random brightness and contrast variation by a random
       linear transform. 
       out_image = k * image + b
    Args:
        k (float): Multiplier
        b (int): Addendum
    """

    def __init__(self, k=0.0, b=0.0):
        assert 0.0 <= k < 1, "k should be in [0..1)"
        assert 0 <= b < 127, "b should be in [0..127)"
        self.k = k
        self.b = b

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        image = image.astype(np.float)
        if self.k != 0.0:
            k = np.random.uniform(1.0-self.k, 1.0+self.k)
            image = image * k
        if self.b != 0:
            b = np.random.randint(-self.b, self.b)
            image = image + b
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
