"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import random
import numpy as np
import torch
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, ListConfig
from data import create_dataset
from models import create_model
from util.util import set_gpu_device
from util.visualizer import Visualizer


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def my_override_dirname(overrides: ListConfig) -> str:
    """Process the overrides passed to the app and return a single string"""
    task_overrides: ListConfig = overrides.task
    ret: str = "_".join(task_overrides)
    ret = ret.replace("{", "")
    ret = ret.replace("}", "")
    ret = ret.replace("[", "")
    ret = ret.replace("]", "")
    ret = ret.replace(",", "_")
    ret = ret.replace("/", "_")
    ret = ret.replace("=", "-")
    return ret

OmegaConf.register_new_resolver("my_override_dirname", my_override_dirname)

@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(opt):
    opt.seed = opt.get('seed', None)
    if opt.seed:
        torch_fix_seed(opt.seed)

    if opt.suffix:
        opt.suffix = (opt.suffix.format(**dict(opt))) if opt.suffix != '' else ''

    set_gpu_device(opt)

    opt.output_dir = HydraConfig.get().runtime.output_dir

    opt.dataroot = hydra.utils.to_absolute_path(opt.dataroot)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    for i, data in enumerate(dataset):  # inner loop within one epoch
        print(data["maskA"])
        print(data["maskB"])
        break

if __name__ == '__main__':
    main()
