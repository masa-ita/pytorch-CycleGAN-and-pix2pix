import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np


class MultifileDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert opt.color_spaces, "color_spaces is not defined."
        self.color_spaces = opt.color_spaces

        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        A_sizes = []
        self.A_paths = dict()
        for cs in self.color_spaces:
            self.A_paths[cs] = sorted(make_dataset(self.dir_A, opt.max_dataset_size, cs))
            A_sizes.append(len(self.A_paths[cs]))

        assert len(set(A_sizes)) == 1, f"number of files in {self.dir_A} are not same"
        self.A_size = A_sizes[0]  # get the size of dataset A

        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.opt.input_nc *= len(self.color_spaces)


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        crop_pos_x = random.randint(0, self.opt.load_size - self.opt.crop_size)
        crop_pos_y = random.randint(0, self.opt.load_size - self.opt.crop_size)
        self.transform_A = get_transform(self.opt, 
                                         params={"crop_pos": (crop_pos_x, crop_pos_y),
                                                 "flip":True}, 
                                         grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        A_images = []
        for cs in self.color_spaces:
            A_path = self.A_paths[cs][index % self.A_size]  # make sure index is within then range
            A_images.append(np.array(self.transform_A(Image.open(A_path).convert('RGB'))))
        A = np.concatenate(A_images, 0)

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
