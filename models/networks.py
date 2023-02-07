import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='batch'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


##############################################################################
# Encoders and Decoder of the Generator and the Prototype Discriminator
##############################################################################
def define_MHEncoder(n_input_modal, input_nc, ngf, norm='batch', use_dropout=False,
               init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Create the multi-head(multi-branch) encoders of the generator
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = MultiHeadResnetEncoder(n_input_modal, input_nc, ngf, norm_layer=norm_layer,
                                   use_dropout=use_dropout, n_blocks=6)
    return init_net(net, init_type, init_gain, gpu_ids)

class MultiHeadResnetEncoder(nn.Module):

    def __init__(self, n_input_modal, input_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct the Multi-Head encoders
        """
        assert (n_blocks >= 0)
        super(MultiHeadResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.n_input_modal = n_input_modal
        n_downsampling = 2
        self.n_downsampling = n_downsampling

        for encoder_idx in range(n_input_modal):
            model = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]
            self.add_module('encoder_{}_0'.format(encoder_idx), nn.Sequential(*model))
            for i in range(n_downsampling):  # add downsampling layers
                mult = 2 ** i
                model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                         norm_layer(ngf * mult * 2),
                         nn.ReLU(True)]
                self.add_module('encoder_{}_{}'.format(encoder_idx, i + 1), nn.Sequential(*model))

        self.dim_reduce_conv = nn.Conv2d(ngf * 4 * n_input_modal, ngf * 4, kernel_size=1)
        self.relu = nn.ReLU(True)

        decoder = []
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            decoder += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        """Standard forward"""
        inputs = torch.chunk(input, self.n_input_modal, dim=1)
        encoder_features = []
        for i in range(self.n_input_modal):
            encoder_features.append([])
            x = inputs[i]
            for j in range(self.n_downsampling + 1):
                x = self.__getattr__('encoder_{}_{}'.format(i, j))(x)
                encoder_features[-1].append(x)
        modal_features = [f[-1] for f in encoder_features]
        features = torch.cat(modal_features, dim=1)
        fused_features = self.dim_reduce_conv(features)
        fused_features = self.relu(fused_features)
        out = self.decoder(fused_features)

        return out


def define_Decoder(output_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Construct the decoder of the generator
    """
    norm_layer = get_norm_layer(norm_type=norm)
    net = Decoder(output_nc, ndf, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)


class Decoder(nn.Module):

    def __init__(self, output_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        Construct the decoder of the generator
        """
        super(Decoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        decoder = []

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            decoder += [nn.ConvTranspose2d(ndf * mult, int(ndf * mult / 2),
                                           kernel_size=3, stride=2,
                                           padding=1, output_padding=1,
                                           bias=use_bias),
                        norm_layer(int(ndf * mult / 2)),
                        nn.ReLU(True)]
        decoder += [nn.ReflectionPad2d(3)]
        decoder += [nn.Conv2d(ndf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        return self.decoder(x)


def define_D(input_nc, ndf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], n_cls=None):
    """Create the Prototype discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = PrototypeDiscriminator(input_nc, ndf, norm_layer=norm_layer, n_cls=n_cls)
    return init_net(net, init_type, init_gain, gpu_ids)


class PrototypeDiscriminator(nn.Module):
    """Defines the Prototype discriminator"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, n_cls=None):
        """Construct the Prototype discriminator
        """
        super(PrototypeDiscriminator, self).__init__()
        self.centroid_size = 512
        self.model_output_size = 512
        self.gamma = 0.999
        self.length_scale = 30
        self.W = nn.Parameter(
            torch.normal(torch.zeros(self.centroid_size, n_cls, self.model_output_size), 0.05)
        )

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ndf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ndf),
                 nn.ReLU(True)]
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(ndf * mult * 2),
                     nn.ReLU(True)]
        self.model = nn.Sequential(*model)

        self.patch_branch = nn.Conv2d(ndf * 4, 1, kernel_size=8, stride=2, padding=1)
        sequence = [nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                     norm_layer(ndf * 4),
                     nn.ReLU(True)]
        self.conv = nn.Sequential(*sequence)
        self.cls_branch = nn.Conv2d(ndf * 4, self.model_output_size, kernel_size=32, stride=1, padding=0)

        self.register_buffer("N", torch.zeros(n_cls) + 1)
        self.register_buffer(
            "m", torch.normal(torch.zeros(self.centroid_size, n_cls), 0.05)
        )
        self.m = self.m * self.N.unsqueeze(0)
        self.sigma = nn.Parameter(torch.zeros(n_cls) + self.length_scale)

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        embeddings = self.m / self.N.unsqueeze(0)
        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()
        return diff

    def forward(self, input):
        """Standard forward."""
        features = self.model(input)
        pred_fake = self.patch_branch(features)
        fe = self.conv(features)
        cls = self.cls_branch(fe)
        b, c, _, _ = cls.shape
        cls = cls.view([b, c])
        y_pred = self.rbf(cls)
        return pred_fake, y_pred, features


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out