# CS_IOC5008_0856078_HW2
# Note
***dataroot:*** I put the CelebA dataset in "data/img_align_celeba" folder  
***HW2_wgan_gp.py:*** Main function of WGAN with gradient penalty  
***HW2_wgan_gp.py:*** Main function of DCGAN  
***model_conGD.py:*** CNN moedel of discriminator and generator  
***model_resGD.py:*** ResNet moedel of discriminator and generator  
***model_linearGD.py:*** FCN moedel of discriminator and generator  
***model_dcgan.py:*** DCGAN moedel of discriminator and generator  
***data_preprocessing.py:*** Output image in 3 x 3 grid  

# Brief introduction:
　The task of this homework is generating new data from CelebA dataset. There are 202599 face images in this dataset. I trained a Wasserstein Generative Adversarial Networks (WGAN) with gradient penalty (GP) to complete this competition. Moreover, I also made use of cropping image, tuning hyperparameters, and changing model architecture to improve performance.
 
# Methodology:
　I choose normal convolutional neural network and Resnet as my backbone of discriminator and generator respectively. I also try to apply Resnet on discriminator, but the performance is worse and take more time. And the loss function is from WGAN-GP. It is more stable than DCGAN. Besides, I also do data preprocessing. I cropped the image from the center to 130 (height) x 60 (width) size. Then it was resized to 64 x 64. Finally, I followed the most hyperparameter setting from WGAN-GP paper. (optimizer = Adam, learning rate = 0.0001, beta1 = 0, beta2 = 0.9, latent size = 128, lambda of GP = 10, iteration ratio of discriminator and generator = 5, batch size = 64)
 
# Findings or Summary:
　In this homework, I use DCGAN first. And I follow most advices in GAN-hacks website like label smoothing, BatchNorm, and so on. However, the generated images mostly look like paintings not realistic images, and are often even not like faces. Then I decide to use WGAN-GP. The result is better but not enough, so I cropped image, tuned hyperparameters, and changed model architecture to improve performance. I will discuss them respectively. The output images shown in this part pass through about 17 epochs  
 
***Cropping image***  
　This is most significant part. It can improve performance very much. At first, I don’t crop the images, and the result is very bad. I think it is because there are some noises in the training dataset. Top part of some images is blurred. Moreover, model can only focus on face not hair or background. It is easier for network to learn. The inspiration of cropping image is by seeing baseline example and results on some papers. I found that the face occupies the almost entire photo, which looks unlike the training images. But, I am not sure whether it is valid in this homework.
Besides, I also found that if we can crop the image, which just fit the face position, we would get better result. However, our output is a square and faces mostly belong to ellipses. Resizing cropped image would make output faces look more like circles as shown in figure 1.   

***Tuning hyperparameter***  
　For the hyperparameters of optimizers, I found that WGAN-GP paper (learning rate = 0.0001, beta1 = 0, beta2 = 0.9) is better than pytorch website (learning rate = 0.0002, beta1 = 0.5, beta2 = 0.999). I also tune the iteration ratio of discriminator and generator (n_critic). I believe that if it is equal to 1 or 2, the results are bad. And the values equal to 4 and 5 have similar result, but I think n_critic equal to 4 is better. I also change the latent size which is input of generator. However, the results are not significantly different.  
 
***Changing model architecture***  
　I found that if we use fully-connected network as backbone of GAN, the resolution of image is very small. The result is so blurred. However, I think we can still identify the face in the images. And what make me surprise is that FCN is better than CNN for learning facial contour. I can identify the face in all output images from FCN, but CNN sometimes generates very weird images. I also try to apply Resnet on both, but the performance is worse. I think this is because the task of discriminator is easier than generator. Powerful discriminator makes training generator harder. Therefore, if the model of generator is more powerful than the other, we can balance both strengths.

# Acknowledgements
https://github.com/eriklindernoren/PyTorch-GAN  
https://github.com/jalola/improved-wgan-pytorch  
https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch  
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
