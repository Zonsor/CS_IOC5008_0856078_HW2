# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:49:32 2019

@author: Zonsor
"""
import os
import numpy as np
import helper
import matplotlib.pyplot as plt
import torch


def output_fig(images_array, file_name="./results"):
    # the shape of your images_array should be (9, width, height, 3),
    # 28 <= width, height <= 112
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(helper.images_square_grid(images_array))
    plt.axis("off")
    plt.savefig(file_name+'.png', bbox_inches='tight', pad_inches=0)


def generate_final_result(genererator):
    os.makedirs("final_result", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(500):
        z_vetcor = torch.randn(9, 128, device=device)
        generated_images = genererator(z_vetcor).detach().cpu()
        generated_images = np.transpose(generated_images, (0, 2, 3, 1))
        output_fig(
                generated_images.numpy(),
                file_name="final_result/{}_image".format(str.zfill(str(i), 3))
                )
