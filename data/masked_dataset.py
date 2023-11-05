import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class MaskedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets with masks.

    It requires four directories to host training images from domain A '/path/to/data/trainA',
     '/path/to/data/train_maskA', and from domain B '/path/to/data/trainB' and
     '/path/to/data/train_maskB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA', '/path/to/data/testB' and  '/path/to/data/test_maskA',
    '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_maskA = os.path.join(opt.dataroot, opt.phase + '_maskA')  # create a path '/path/to/data/train_maskA'
        self.dir_maskB = os.path.join(opt.dataroot, opt.phase + '_maskB')  # create a path '/path/to/data/train_maskB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            maskA (tensor)
            maskB (tensor)
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        maskA_path = os.path.join(self.dir_maskA, os.path.basename(A_path))
        maskB_path = os.path.join(self.dir_maskB, os.path.basename(B_path))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        maskA_img = Image.open(maskA_path).convert('RGB')
        maskB_img = Image.open(maskB_path).convert('RGB')
        
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        crop_pos_x = random.randint(0, self.opt.load_size - self.opt.crop_size)
        crop_pos_y = random.randint(0, self.opt.load_size - self.opt.crop_size)
        self.transform_A = get_transform(self.opt, 
                                         params={"crop_pos": (crop_pos_x, crop_pos_y),
                                                 "flip":True}, 
                                         grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, 
                                         params={"crop_pos": (crop_pos_x, crop_pos_y),
                                                 "flip":True}, 
                                         grayscale=(output_nc == 1))
        self.transform_maskA = get_transform(self.opt,
                                             params={"crop_pos": (crop_pos_x, crop_pos_y),
                                                 "flip":True}, 
                                             grayscale=(input_nc == 1),
                                             mask=True)
        self.transform_maskB = get_transform(self.opt, 
                                             params={"crop_pos": (crop_pos_x, crop_pos_y),
                                                 "flip":True}, 
                                             grayscale=(output_nc == 1), mask=True)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        maskA = self.transform_maskA(maskA_img)
        maskB = self.transform_maskB(maskB_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'maskA': maskA, 'maskB': maskB}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
