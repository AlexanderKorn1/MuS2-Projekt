import argparse
import os
import numpy as np
import math
import sys
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

# make directory for saving the generated images
os.makedirs("images", exist_ok=True)

# parse user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=10, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=5, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

# define image shape
img_shape = (opt.channels, opt.img_size, opt.img_size)

# check for nvidia gpu, true if cuda driver is available
cuda = True if torch.cuda.is_available() else False

# class definition - Generator
class Generator(nn.Module):
    def __init__(self):                                             
        super(Generator, self).__init__()                           # constructor of Generator

        def block(in_feat, out_feat, normalize=True):               # define implement layers of Generator
            layers = [nn.Linear(in_feat, out_feat)]                 # set linear layer
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))        # append normalize first layer
            layers.append(nn.LeakyReLU(0.2, inplace=True))          # append LeakyReLu layers
            return layers                                           # return layers

        self.model = nn.Sequential(                                 # define Generator model
            *block(opt.latent_dim, 128, normalize=False),           # define convolutions
            *block(128, 256),                                       
            *block(256, 512),                                       
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),               # applies linear transformation to incoming data
            nn.Tanh()                                               # tangens hyperbolicus
        )

    def forward(self, z):                                           # define help method
        img = self.model(z)                                         # prepare image
        img = img.view(img.shape[0], *img_shape)                    # prepare image shape
        return img


# class definition - Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()                       # constructor of Discriminator

        self.model = nn.Sequential(                                 # define Discriminator model
            nn.Linear(int(np.prod(img_shape)), 512),                # define convolution layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):                                         # define help method
        img_flat = img.view(img.shape[0], -1)                       
        validity = self.model(img_flat)
        return validity


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# if cuda is available, use gpu
if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
transform = transforms.Compose([transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])    # define transformation
dataset = datasets.ImageFolder('pics', transform=transform)         # path to input dataset and transform
dataloader = torch.utils.data.DataLoader(                           # load dataset
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)            # optimizer for Generator
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)        # optimizer for Discriminator

# prepare Torch Tensor or Cuda Tensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):                                   # iterating through the defined epochs

    for i, (imgs, _) in enumerate(dataloader):                      # iterating through dataset

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))                     # prepare input real image

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # set all gradients of Tensors to zero
        optimizer_D.zero_grad()                                     

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

        # compute gradient
        loss_D.backward()

        # perform optimizing step
        optimizer_D.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # set all gradients of Tensors to zero
            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)

            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_imgs))

            # compute gradient
            loss_G.backward()

            # perform optimizing step
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, loss_D.item(), loss_G.item())
            )

        # save generated image
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:1], "images/%d.png" % batches_done, nrow=1, normalize=True)
        # count done batches up
        batches_done += 1
