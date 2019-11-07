# CS_IOC5008_0856078_HW2
# Brief introduction:
　The task of this homework is generating new data from CelebA dataset. There are 202599 face images in this dataset. I trained a Wasserstein Generative Adversarial Networks (WGAN) with gradient penalty (GP) to complete this competition. Moreover, I also made use of cropping image, tuning hyperparameters, and changing model architecture to improve performance. Then I get final result as shown in figure 1 (compared to the real face image as shown in figure 2.).
# Methodology:
　I choose normal convolutional neural network and Resnet as my backbone of discriminator and generator respectively. The architecture is shown in table 1. I also try to apply Resnet on discriminator, but the performance is worse and take more time. And the loss function is from WGAN-GP. It is more stable than DCGAN. Besides, I also do data preprocessing. I cropped the image from the center to 130 (height) x 60 (width) size. Then it was resized to 64 x 64. Finally, I followed the most hyperparameter setting from WGAN-GP paper. (optimizer = Adam, learning rate = 0.0001, beta1 = 0, beta2 = 0.9, latent size = 128, lambda of GP = 10, iteration ratio of discriminator and generator = 5, batch size = 64)

# Acknowledgements
https://github.com/eriklindernoren/PyTorch-GAN  
https://github.com/jalola/improved-wgan-pytorch  
https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch  
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
