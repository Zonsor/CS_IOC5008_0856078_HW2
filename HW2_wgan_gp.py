# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:44:14 2019

@author: Zonsor
"""

import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.autograd as autograd
import torch

import torchvision.utils as vutils
import matplotlib.pyplot as plt
import model_conGD
import model_resGD
import model_linearGD
from data_processing import output_fig, generate_final_result


def plot_training_image(dataloader):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    a = np.transpose(vutils.make_grid(real_batch[0][:9],
                                      padding=2, normalize=True).cpu(), (1, 2, 0))
    output_fig(a, file_name="./training_img")
    plt.imshow()


os.makedirs("images", exist_ok=True)

n_epochs = 20
batch_size = 64
lr = 0.0001
b1 = 0.
b2 = 0.9
latent_dim = 128
img_size = 64
channels = 3
n_critic = 4
clip_value = 0.01
sample_interval = 400
ngpu = 1
ngf = 64
ndf = 64
model_g = "res"  # linear, conv, or res
model_d = "conv"  # linear, conv, or res
# what kind of cropped you want?
# 1 -> same width and height
# 2 -> different width and height
cropped_confi = 2

img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
if model_g == "linear":
    generator = model_linearGD.Generator()
elif model_g == "conv":
    generator = model_conGD.Generator(input_dim=latent_dim, output_channels=channels, dim=img_size)
else:
    generator = model_resGD.Generator()

if model_g == "linear":
    discriminator = model_linearGD.Discriminator()
elif model_g == "conv":
    discriminator = model_conGD.Discriminator(input_channels=channels, dim=img_size, norm='layer_norm')
else:
    discriminator = model_resGD.Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# =======================================================
# ==                      dataset                      ==
# =======================================================
dataroot = "data/"

if cropped_confi == 2:
    crop_size_h = 130
    crop_size_w = 86
    offset_height = (218 - crop_size_h) // 2
    offset_width = (178 - crop_size_w) // 2
    crop = lambda x: x[:, offset_height+5:offset_height+5 + crop_size_h, offset_width:offset_width + crop_size_w]
else:
    crop_size = 108
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height+5:offset_height+5 + crop_size, offset_width:offset_width + crop_size]

dataset = datasets.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(crop),
                                transforms.ToPILImage(),
                                transforms.Resize(size=(img_size, img_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

#######################################################

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples.detach() + ((1 - alpha) * fake_samples.detach())).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

fixed_noise = Variable(Tensor(np.random.normal(0, 1, (25, latent_dim))))

print("Starting Training Loop...")
batches_done = 0
for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs.detach())
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

        if batches_done % 10 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %.4f] [G loss: %.4f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        if batches_done % sample_interval == 0:
            with torch.no_grad():
                test = generator(fixed_noise).detach().cpu()
            save_image(test.data, "images/%d.png" % batches_done, nrow=5, normalize=True)

        batches_done += 1

torch.save(generator.state_dict(), 'pkl/WGAN_GP_G.pkl')
torch.save(discriminator.state_dict(), 'pkl/WGAN_GP_D.pkl')

generate_final_result(generator)
