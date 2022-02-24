# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np
import torch
torch.manual_seed(0)
import os
device_ids = [0]
from PIL import Image
import torch.nn as nn
import re
import base64
import torch.optim as optim
import  math
import cv2
import gc
import torch.nn.functional as F
from tqdm import tqdm
import copy
from models.stylegan2_pytorch.stylegan2_pytorch import Generator as Stylegan2Generator
from models.stylegan1_pytorch.stylegan1 import G_mapping, Truncation, G_synthesis

from models.DatasetGAN.classifer import pixel_classifier
from collections import OrderedDict
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def latent_to_image(g_all, upsamplers, latents, return_upsampled_layers=False, use_style_latents=False,
                    process_out=True, return_stylegan_latent=False, dim=512,
                    return_only_im=False, noise=None):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''

    # assert (len(latents) == 1)  # for GPU memory constraints
    if not use_style_latents:
        # generate style_latents from latents
        style_latents = g_all.module.truncation(g_all.module.g_mapping(latents))
        style_latents = style_latents.clone()  # make different layers non-alias

    else:
        style_latents = latents

    if return_stylegan_latent:
        return  style_latents

    img_list, affine_layers = g_all.module.g_synthesis(style_latents, noise=noise)


    if return_only_im:
        if process_out:
            if img_list.shape[-2] > 512:
                img_list = upsamplers[-1](img_list)
            img_list = img_list.cpu().detach().numpy()
            img_list = process_image(img_list)
            img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        return img_list, style_latents

    number_feautre = 0

    for item in affine_layers:
        number_feautre += item.shape[1]

    if return_upsampled_layers:
        affine_layers_upsamples = torch.FloatTensor(1, number_feautre, dim, dim).cuda()
        start_channel_index = 0
        for i in range(len(affine_layers)):
            len_channel = affine_layers[i].shape[1]
            affine_layers_upsamples[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](
                affine_layers[i])
            start_channel_index += len_channel
    else:
        affine_layers_upsamples = affine_layers

    if img_list.shape[-2] != 512:
        img_list = upsamplers[-1](img_list)

    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)

    return img_list, affine_layers_upsamples


def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)


def reverse_process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images - (0.5 - drange[0] * scale)
    images = images / scale

    return images

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    # t > .75
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    # t < 0.05
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


class Interpolate(nn.Module):
    def __init__(self, size, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.align_corners:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, size=self.size, mode=self.mode)
        return x




def run_embedding_optimization(args, g_all, upsamplers, inter, percept, img_tensor, latent_in,
                               steps=1000, stylegan_encoder=None, regular_by_org_latent=False, early_stop=False,
                               encoder_loss_weight=1, use_noise=False, noise_loss_weight=100):
    gc.collect()
    torch.cuda.empty_cache()
    latent_in = latent_in.detach()
    img_tensor = (img_tensor + 1.0) / 2.0

    org_latnet_in = copy.deepcopy(latent_in.detach())
    with torch.no_grad():
        im_out_wo_opti, _ = latent_to_image(g_all, upsamplers, org_latnet_in, process_out=False,
                                            dim=args['im_size'][1],
                                            use_style_latents=True, return_only_im=True)

        im_out_wo_opti = inter(im_out_wo_opti)
        im_out_wo_opti = (im_out_wo_opti + 1.0) / 2.0
        p_loss = percept(im_out_wo_opti, img_tensor).mean()
        mse_loss = F.mse_loss(im_out_wo_opti, img_tensor)

    best_loss = args['loss_dict']['p_loss'] * p_loss + \
                args['loss_dict']['mse_loss'] * mse_loss

    if args['truncation']:
        latent_in = g_all.module.truncation(latent_in)

    latent_in.requires_grad = True

    if use_noise:
        noises = g_all.module.make_noise()

        for noise in noises:
            noise.requires_grad = True
    else:
        noises = None
    if not use_noise:
        optimizer = optim.Adam([latent_in], lr=3e-5)
    else:

        optimizer = optim.Adam([latent_in] + noises, lr=3e-5)
        optimized_noise = noises

    count = 0
    optimized_latent = latent_in

    loss_cache = [best_loss.item()]
    for _ in tqdm(range(1, steps)):
        t = _ / steps
        lr = get_lr(t, 0.1)
        optimizer.param_groups[0]['lr'] = lr

        img_out, _ = latent_to_image(g_all, upsamplers, latent_in, process_out=False,
                                     dim=args['im_size'][1],
                                     use_style_latents=True, return_only_im=True, noise=noises)

        img_out = inter(img_out)

        img_out = (img_out + 1.0) / 2.0



        p_loss = percept(img_out, img_tensor).mean()

        mse_loss = F.mse_loss(img_out, img_tensor)
        if regular_by_org_latent:
            encoder_loss = F.mse_loss(latent_in, org_latnet_in)
        else:
            encoder_loss = F.mse_loss(latent_in, stylegan_encoder(img_out).detach())

        reconstruction_loss = args['loss_dict']['p_loss'] * p_loss + \
                              args['loss_dict']['mse_loss'] * mse_loss

        loss = reconstruction_loss + encoder_loss_weight * encoder_loss


        if use_noise:
            n_loss = noise_regularize(noises)

            loss += noise_loss_weight * n_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cache.append(reconstruction_loss.item())
        if reconstruction_loss.item() < best_loss:
            best_loss = reconstruction_loss.item()
            count = 0
            optimized_latent = latent_in.detach()
            if use_noise:
                optimized_noise = [noise.detach().cpu().numpy() for noise in noises]
        else:
            count += 1
        if early_stop and count > 100:
            break


    gc.collect()
    torch.cuda.empty_cache()

    if use_noise:
        return optimized_latent, optimized_noise, loss_cache
    else:
        return optimized_latent, None, loss_cache


def prepare_model(args, classfier_checkpoint_path="", classifier_iter=10000, num_class=34, num_classifier=10):

    if args['category'] == 'face' or args['category'] == 'flickr_car':
        res = 1024
        out_res = 512
    elif args['category'] == 'face_256':
        res = 256
        out_res = 256

    else:
        res = 512
        out_res = 512

    if args['stylegan_ver'] == "1":

        if args['category'] == "car":
            max_layer = 8
        elif args['category'] == "face":
            max_layer = 8
        elif args['category'] == "bedroom":
            max_layer = 7
        elif args['category'] == "cat":
            max_layer = 7
        else:
            assert "Not implementated!"


        avg_latent = np.load(args['average_latent'])
        avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).cuda()

        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=res))
        ]))

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()

    elif args['stylegan_ver'] == "2":
        g_all = Stylegan2Generator(res, 512, 8, channel_multiplier=2, randomize_noise=False)
        checkpoint = torch.load(args['stylegan_checkpoint'])

        print("Load stylegan from, " , args['stylegan_checkpoint'], " at res, ", str(res))
        g_all.load_state_dict(checkpoint["g_ema"], strict=True)
        avg_latent = g_all.make_mean_latent(4086)


    g_all.eval()
    g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()
    mode = "nearest"
    nn_upsamplers = [nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode)]

    if res > 512:
        nn_upsamplers.append(Interpolate(512, mode, align_corners=None))
        nn_upsamplers.append(Interpolate(512, mode, align_corners=None))


    mode = 'bilinear'
    bi_upsamplers = [nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode)]

    if res > 512:
        bi_upsamplers.append(Interpolate(512, mode))
        bi_upsamplers.append(Interpolate(512, mode))

    if classfier_checkpoint_path != "":
        print("Load Classifier path, ", classfier_checkpoint_path)
        classifier_list = []
        for MODEL_NUMBER in range(num_classifier):
            classifier = pixel_classifier(num_class, dim=args['dim'])
            classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
            if classifier_iter > 0:
                checkpoint = torch.load(os.path.join(classfier_checkpoint_path,
                                                     'car_model_iter' + str(classifier_iter) + '_number_' + str(
                                                         MODEL_NUMBER) + '.pth'))
            else:
                checkpoint = torch.load(os.path.join(classfier_checkpoint_path,
                                                     'car_model_' + str(MODEL_NUMBER) + '.pth'))

            classifier.load_state_dict(checkpoint['model_state_dict'], strict=True)
            classifier.eval()
            classifier_list.append(classifier)

        for c in classifier_list:
            for i in c.parameters():
                i.requires_grad = False
    else:
        classifier_list = []

    for i in g_all.parameters():
        i.requires_grad = False



    return g_all, nn_upsamplers, bi_upsamplers, classifier_list, avg_latent
