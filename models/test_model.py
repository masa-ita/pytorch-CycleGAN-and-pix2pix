from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake', 'mask', 'masked_real', 'masked_fake', 'combined']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        if self.model_names[0] == "G_A":
            self.real = input['A'].to(self.device)
            self.image_paths = input['A_paths']
            self.mask = input['maskA'].to(self.device)
        elif self.model_names[0] == "G_B":
            self.real = input['B'].to(self.device)
            self.image_paths = input['B_paths']
            self.mask = input['maskB'].to(self.device)
        else:
            raise ValueError("invalid model_suffix")
        self.masked_real = self.real * self.mask
        
    def forward(self):
        """Run forward pass."""
        self.fake = self.netG(self.real)  # G(real)
        self.masked_fake = self.fake * self.mask
        self.combined = self.masked + self.real * (1 - self.mask)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass
