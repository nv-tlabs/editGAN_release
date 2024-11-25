# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import numpy as np
import torch
import random
import torch.nn as nn
import pickle
torch.manual_seed(0)
import scipy.misc
import json
import torch.nn.functional as F
import os
import imageio

device_ids = [0]
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from models.encoder.encoder import FPNEncoder

from utils.data_utils import *
from utils.model_utils import *

import torch.optim as optim
import argparse
import glob
import lpips

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def embed_one_example(args, path, stylegan_encoder, g_all, upsamplers,
                      inter, percept, steps, sv_dir,
                      skip_exist=False):

    if os.path.exists(sv_dir):
        if skip_exist:
            return 0,0,[], []
        else:
            pass
    else:
        os.system('mkdir -p %s' % (sv_dir))
    print('SV folder at: %s' % (sv_dir))
    image_path = path
    label_im_tensor, im_id = load_one_image_for_embedding(image_path, args['im_size'])

    print("****** Run optimization for ", path, " ******")


    label_im_tensor = label_im_tensor.to(device)
    label_im_tensor = label_im_tensor * 2.0 - 1.0
    label_im_tensor = label_im_tensor.unsqueeze(0)
    latent_in = stylegan_encoder(label_im_tensor)
    with torch.no_grad():
        im_out_wo_encoder, _ = latent_to_image(g_all, upsamplers, latent_in,
                                           process_out=True, use_style_latents=True,
                                           return_only_im=True)

    out = run_embedding_optimization(args, g_all,
                                     upsamplers, inter, percept,
                                     label_im_tensor, latent_in, steps=steps,
                                     stylegan_encoder=stylegan_encoder,
                                     use_noise=args['use_noise'],
                                     noise_loss_weight=args['noise_loss_weight']
                                     )
    if args['use_noise']:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = [torch.from_numpy(noise).cuda() for noise in optimized_noise]
    else:
        optimized_latent, optimized_noise, loss_cache = out
        optimized_noise = None
    print("Curr loss, ", loss_cache[0], loss_cache[-1] )

    optimized_latent_np = optimized_latent.detach().cpu().numpy()[0]
    if args['use_noise']:
        loss_cache_np = [noise.detach().cpu().numpy() for noise in optimized_noise]
    else:
        loss_cache_np = []
    # vis
    img_out, _ = latent_to_image(g_all, upsamplers, optimized_latent,
                                 process_out=True, use_style_latents=True,
                                 return_only_im=True, noise=optimized_noise)

    raw_im_show = (np.transpose(label_im_tensor.cpu().numpy(), (0, 2, 3, 1))) * 255.

    vis_list = [im_out_wo_encoder[0], img_out[0]
                ]

    curr_vis = np.concatenate(
        vis_list, 0)
    imageio.imsave(os.path.join(sv_dir, "reconstruction.jpg"),
                      curr_vis)

    imageio.imsave(os.path.join(sv_dir, "real_im.jpg"),
                      raw_im_show[0])

    return loss_cache[0], loss_cache[-1], optimized_latent_np, loss_cache_np

def test(args, resume, steps,  latent_sv_folder='', skip_exist=False):

    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    inter = Interpolate(args['im_size'][1], 'bilinear')

    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), normalize=args['normalize']
    ).to(device)
    stylegan_encoder = FPNEncoder(3, n_latent=args['n_latent'], only_last_layer=args['use_w'])
    stylegan_encoder = stylegan_encoder.to(device)
    stylegan_encoder.load_state_dict(torch.load(resume, map_location=device)['model_state_dict'], strict=True)

    assert latent_sv_folder != ""
    all_images = []
    all_id = []

    curr_images_all = glob.glob(args['testing_data_path'] +  "*/*")
    curr_images_all = [data for data in curr_images_all if ('jpg' in data or 'webp' in data or 'png' in data  or 'jpeg' in data or 'JPG' in data) and not os.path.isdir(data)  and not 'npy' in data ]

    for i, image in enumerate(curr_images_all):
        all_id.append(image.split("/")[-1].split(".")[0])
        all_images.append(image)

    print("All files, " , len(all_images))


    all_loss_before_opti, all_loss_after_opti = [], []
    for i, id in enumerate(tqdm(all_id)):

        print("Curr dir,", id)

        sv_folder = os.path.join(latent_sv_folder,  id, 'crop_latent_' + str(steps))

        loss_before_opti, loss_after_opti , all_final_latent, all_final_noise = embed_one_example(args, all_images[i],
                                                                                                  stylegan_encoder, g_all,
                                                                                                  upsamplers, inter, percept, steps,
                                                                                                  sv_folder, skip_exist=skip_exist)

        all_loss_before_opti.append(loss_before_opti)
        all_loss_after_opti.append(loss_after_opti)

        id_num = id.split("_")[-1]
        latent_name = latent_sv_folder +  '/latents_image_%s.npy' % str(id_num)
        np.save(latent_name, all_final_latent)

        if len(all_final_noise) != 0:

            latent_name = latent_sv_folder + '/latents_image_%s_npose.npy' % str(id_num)

            with open(latent_name, 'wb') as handle:
                pickle.dump(all_final_noise, handle)

    result = {"before:": np.mean(all_loss_before_opti), "after": np.mean(all_loss_after_opti)}

    with open(latent_sv_folder + '/result.json', 'w') as f:
        json.dump(result, f)


def main(args, resume):

    base_path = os.path.join(args['exp_dir'], "checkpoint")
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    percept = lpips.PerceptualLoss(
        model='net-lin', net='vgg',
        use_gpu=device.startswith('cuda'), normalize=args['normalize']
    ).to(device)
    stylegan_encoder = FPNEncoder(3, n_latent=args['n_latent'], only_last_layer=args['use_w'], same_view_code=args['same_view_code'])
    stylegan_encoder = stylegan_encoder.to(device)
    stylegan_encoder.eval()
    if resume != "":
        stylegan_encoder.load_state_dict(torch.load(resume, map_location=device)['model_state_dict'])

    optimizer = optim.Adam(stylegan_encoder.parameters(), lr=args['lr'])
    inter = Interpolate(args['im_size'][1], 'bilinear')
    images_all = glob.glob(args['training_data_path'] + "/*")

    images_all = [data for data in images_all if 'jpg' in data or 'webp' in data or 'png' in data]
    if args['debug']:
        images_all = images_all[:1]
    print( "Training data length, ", len(images_all))

    if 'car' in args['category']:
        fill_blank = True
    else:
        fill_blank = False

    train_data = trainData(images_all, img_size=args['im_size'], fill_blank=fill_blank)
    shuffle = True
    if args['debug']:
        args['bs'] = 1
        shuffle = False

    train_data_loader = DataLoader(train_data, batch_size=args['bs'], shuffle=shuffle, num_workers=8)

    if args['debug']:
        np.random.seed(41)
        sampling_latent = np.random.randn(1, 512)

    best_loss = 999999999
    for epoch in range(100):

        for i, da, in enumerate(train_data_loader):
            if i % 10 == 0:
                gc.collect()
            img_tensor = da[0].to(device)
            img_tensor = img_tensor  *  2.0 - 1.0
            stylegan_encoder.train()
            latent_in = stylegan_encoder(img_tensor)
            if args['truncation']:
                latent_in = g_all.module.truncation(latent_in)

            img_out , _ = latent_to_image(g_all, upsamplers, latent_in, use_style_latents=True,
                                          return_only_im=True, process_out=False)


            img_out = inter(img_out)
            img_tensor = (img_tensor + 1.0) / 2.0
            img_out = (img_out + 1.0) / 2.0

            p_loss = percept(img_out, img_tensor).mean()
            mse_loss = F.mse_loss(img_out, img_tensor)

            loss = p_loss * args['loss_dict']['p_loss'] + \
                   mse_loss * args['loss_dict']['mse_loss']

            if epoch > args['train_real_start_epochs']:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if args['sampling_training']:

                sampling_loss = 0
                sample_img = []
                sample_latnet = []
                for round in range(args['sample_bs']):
                    if args['debug']:
                        latent = sampling_latent

                    else:
                        latent = np.random.randn(1, 512)
                    latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)

                    with torch.no_grad():
                        curr_sample_img, curr_sample_latnet = latent_to_image(g_all, upsamplers, latent,
                                                             return_only_im=True, process_out=False)

                    sample_img.append(curr_sample_img)
                    sample_latnet.append(curr_sample_latnet)

                sample_img = torch.cat(sample_img, 0)

                sample_img = inter(sample_img)

                sample_latnet = torch.cat(sample_latnet, 0)

                encode_latent = stylegan_encoder(sample_img)

                encoder_loss = F.mse_loss(encode_latent, sample_latnet)

                sampling_loss += encoder_loss * args['loss_dict']['encoder_loss']

                if args['truncation']:
                    encode_latent = g_all.module.truncation(encode_latent)

                img_out_sampling, _ = latent_to_image(g_all, upsamplers, encode_latent, use_style_latents=True, return_only_im=True, process_out=False)
                img_out_sampling = inter(img_out_sampling)

                img_out_sampling = (img_out_sampling + 1.0) / 2.0
                sample_img = (sample_img + 1.0) / 2.0
                p_loss = percept(img_out_sampling, sample_img).mean()
                mse_loss = F.mse_loss(img_out_sampling, sample_img)
                sampling_loss += p_loss * args['loss_dict']['p_loss'] + \
                       mse_loss * args['loss_dict']['mse_loss']
                optimizer.zero_grad()
                sampling_loss.backward()
                optimizer.step()

            if args['debug']:
                image_name = os.path.join(args['exp_dir'], 'real.png')

                img_out = 255 * np.transpose( (img_out.detach().cpu().numpy()[0]), (1,2,0))
                img_gt = 255 * np.transpose( (img_tensor.cpu().numpy()[0]), (1,2,0))

                img_save = (np.concatenate( (img_out, img_gt), 1)).astype(np.uint8)
                img_out = Image.fromarray(img_save)
                img_out.save(image_name)

                if args['sampling_training']:
                    image_name = os.path.join(args['exp_dir'], 'sampling.png')


                    img_out = 255 * np.transpose((img_out_sampling.detach().cpu().numpy()[0]), (1, 2, 0))
                    img_gt = 255 * np.transpose((sample_img.detach().cpu().numpy()[0]), (1, 2, 0))

                    img_save = (np.concatenate((img_out, img_gt), 1)).astype(np.uint8)
                    img_out = Image.fromarray(img_save)
                    img_out.save(image_name)
                exit()
            if i % 10 == 0 :
                if args['sampling_training']:
                    print(epoch, 'epoch', 'iteration', i, 'loss: ', loss.item(), " sampling loss: ", sampling_loss.item())
                else:
                    print(epoch, 'epoch', 'iteration', i, 'loss', loss)


            if i % 2000 == 0 and i!=0 and not args['debug']:
                image_name = os.path.join(args['exp_dir'], 'Epoch_' + str(epoch)  + "_iter_"+ str(i) + '.png')
                img_out = np.transpose(img_out.detach().cpu().numpy()[0], (1,2,0))
                img_gt = np.transpose(img_tensor.cpu().numpy()[0], (1,2,0))
                img_save=  (255 * np.concatenate( (img_out, img_gt), 0)).astype(np.uint8)
                img_out = Image.fromarray(img_save)
                img_out.save(image_name)
                image_name = os.path.join(args['exp_dir'], 'Epoch_' + str(epoch) + "_iter_" + str(i) + '_sampling.png')
                img_out = 255 * np.transpose((img_out_sampling.detach().cpu().numpy()[0]), (1, 2, 0))
                img_gt = 255 * np.transpose((sample_img.detach().cpu().numpy()[0]), (1, 2, 0))
                img_save = (np.concatenate((img_out, img_gt), 1)).astype(np.uint8)
                img_out = Image.fromarray(img_save)
                img_out.save(image_name)
                loss = loss.item()
                model_path = os.path.join(base_path, 'epoch_' + str(epoch) + '_iter_' + str(i) + '_loss_' + str(loss) + '.pth')
                print('Save to:', model_path)
                torch.save({'model_state_dict': stylegan_encoder.state_dict()},
                           model_path)
                if epoch > 2:
                    if loss < best_loss:
                        best_loss = loss
                        model_path = os.path.join(base_path, 'BEST_' + 'loss' + str(loss) + '.pth')
                        print('Save to:', model_path)
                        torch.save({'model_state_dict': stylegan_encoder.state_dict()},
                                   model_path)

            del loss
            if args['sampling_training']:
                del sampling_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--test', type=bool, default=False)

    parser.add_argument('--testing_path', type=str, default='')
    parser.add_argument('--latent_sv_folder', type=str, default='')
    parser.add_argument('--skip_exist', type=bool, default=False)
    parser.add_argument('--steps', type=int, default=500)

    parser.add_argument('--use_noise', type=bool, default=False)
    parser.add_argument('--noise_loss_weight', type=float, default=100)

    args = parser.parse_args()
    opts = json.load(open(args.exp, 'r'))


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    opts['use_noise'] = args.use_noise
    opts['noise_loss_weight'] = args.noise_loss_weight

    print("Opt", opts)

    if args.testing_path != "":
        opts['testing_data_path'] = args.testing_path

    if args.test:
        test(opts, args.resume, args.steps,
             latent_sv_folder=args.latent_sv_folder, skip_exist=args.skip_exist)
    else:
        main(opts, args.resume)

