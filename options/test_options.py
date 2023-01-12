from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, opt):
        opt = BaseOptions.initialize(self, opt)  # define shared options

        opt.model='test'
        # To avoid cropping, the load_size should be the same as crop_size
        opt.load_size = opt.crop_size
        self.isTrain = False
        return opt
