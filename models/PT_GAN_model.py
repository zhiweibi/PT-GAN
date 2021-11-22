from collections import OrderedDict
import torch
from torch import nn
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from itertools import chain

class PTGANModel(BaseModel):

    def __init__(self, opt):
        """Initialize the PT-GAN.

        Parameters:
            opt -- stores all the experiment flags
        """
        BaseModel.__init__(self, opt)
        self.n_input_modal = opt.n_input_modal
        # specify the training losses you want to print out.
        self.loss_names = ['G_GAN', 'G_L1', 'G_bce', 'D_real', 'D_fake', 'D_bce', 'rec']
        if self.isTrain:
            self.model_names = ['G_EN', 'G_DE', 'D']
        else:  # during test time, only load G
            self.model_names = ['G_EN', 'G_DE']
        self.netG_EN = networks.define_MHEncoder(opt.n_input_modal, opt.input_nc+opt.n_input_modal+1, opt.ngf, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_DE = networks.define_Decoder(opt.output_nc, opt.ngf, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.norm, opt.init_type,
                                          opt.init_gain, self.gpu_ids, len(opt.modal_names))

        if self.isTrain:
            # define loss functions
            self.criterionCls = nn.CrossEntropyLoss()
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(chain(self.netG_EN.parameters(), self.netG_DE.parameters()),
                                                lr=self.opt.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(chain(self.netD.parameters(), self.netG_DE.parameters()),
                                                lr=self.opt.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.all_modal_names = opt.modal_names
        self.n_cls = opt.n_input_modal + 1

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.real_B_no_mask = input['B'][:, :self.opt.input_nc].to(self.device)
        self.modal_names = [i[0] for i in input['modal_names']]

        target_modal_names = input['modal_names'][-1]
        self.real_B_Cls = torch.tensor([self.all_modal_names.index(i) for i in target_modal_names]).to(self.device)

    def forward(self):
        """Run forward pass"""
        self.fake_B = self.netG_DE(self.netG_EN(self.real_A))

    def backward_D(self):
        """Calculate loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B.detach()  # fake_B from generator
        g_pred_fake, _, _ = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(g_pred_fake, False)

        # Real
        if self.opt.lambda_GP > 0:
            self.real_B_no_mask.requires_grad_(True)
        pred_real, cls_real, features = self.netD(self.real_B_no_mask)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        self.loss_D_bce = self.bce_loss_fn(cls_real, y)
        self.loss_D_bce += self.opt.lambda_GP * self.calc_gradient_penalty(self.real_B_no_mask, cls_real)
         # the Reconstructive Regularization is optimized together with the discriminator
        rec = self.netG_DE(features)
        self.loss_rec = self.criterionL1(self.real_B_no_mask, rec) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.opt.lambda_bce * self.loss_D_bce + self.loss_rec) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        g_pred_fake, g_cls_fake, _ = self.netD(self.fake_B)
        self.loss_G_GAN = self.criterionGAN(g_pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B_no_mask) * self.opt.lambda_L1
        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        self.loss_G_bce = self.bce_loss_fn(g_cls_fake, y)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_bce

        self.loss_G.backward()

    def update_embeddings(self):
        y = F.one_hot(self.real_B_Cls, self.n_cls).float()
        self.netD.module.N = self.netD.module.gamma * self.netD.module.N + (1 - self.netD.module.gamma) * y.sum(0)
        features = self.netD.module.model(self.real_B_no_mask)
        fe = self.netD.module.conv(features)
        cls = self.netD.module.cls_branch(fe)
        b, c, _, _ = cls.shape
        z = cls.view([b, c])
        z = torch.einsum("ij,mnj->imn", z, self.netD.module.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)
        self.netD.module.m = self.netD.module.gamma * self.netD.module.m + (1 - self.netD.module.gamma) * embedding_sum

    def bce_loss_fn(self, y_pred, y):
        bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
            self.n_cls * y_pred.shape[0]
        )
        return bce

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D and auto-encoder's weights
        self.real_B_no_mask.requires_grad_(False)
        with torch.no_grad():
            self.netD.eval()
            self.update_embeddings()
        self.netD.train()

    def get_current_visuals(self):
        modal_imgs = []
        for i in range(self.n_input_modal):
            modal_imgs.append(self.real_A[:, i*(self.n_input_modal+1+self.opt.input_nc):i*(self.n_input_modal+1+self.opt.input_nc)+self.opt.input_nc, :, :])
        modal_imgs.append(self.real_B_no_mask)
        visual_ret = OrderedDict()
        for name, img in zip(self.modal_names, modal_imgs):
            visual_ret[name] = img
        visual_ret['fake_' + self.modal_names[-1]] = self.fake_B

        return visual_ret

    def calc_gradients_input(self, x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(self, x, y_pred):
        gradients = self.calc_gradients_input(x, y_pred)
        # L2 norm
        grad_norm = gradients.norm(2, dim=1)
        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty