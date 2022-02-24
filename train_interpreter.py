# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')
import imageio
import torch
import torch.nn as nn
torch.manual_seed(0)
import scipy.misc
import json
import numpy as np
device_ids = [0]
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils.model_utils import *
from utils.data_utils import *
from models.DatasetGAN.classifer import pixel_classifier

import scipy.stats
import torch.optim as optim
import argparse
from utils.data_utils import car_32_palette as palette

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import cv2

class trainData(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


def prepare_data(args, palette):

    g_all, _, upsamplers, _, avg_latent = prepare_model(args)
    if args['optimized_latent_path']['train'][-4:] == ".npy":
        latent_all = np.load(args['optimized_latent_path']['train'])
    else:
        latent_all = []
        for i in range(args['max_training']):
            # quickly resolve id mismatch
            if i >= 28:
                i += 1
            name = 'latents_image_%0d.npy' % i

            im_frame = np.load(os.path.join(args['optimized_latent_path']['train'], name))
            latent_all.append(im_frame)
        latent_all = np.array(latent_all)

    latent_all = torch.from_numpy(latent_all).cuda()


    # load annotated mask
    mask_list = []
    im_list = []
    latent_all = latent_all[:args['max_training']]
    num_data = len(latent_all)

    for i in range(len(latent_all)):

        if i >= args['max_training']:
            break
        name = 'image_mask%0d.npy' % i

        im_frame = np.load(os.path.join( args['annotation_mask_path'] , name))
        mask = np.array(im_frame)
        mask =  cv2.resize(np.squeeze(mask), dsize=(args['dim'][1], args['dim'][0]), interpolation=cv2.INTER_NEAREST)

        mask_list.append(mask)

        im_name = os.path.join( args['annotation_mask_path'], 'image_%d.jpg' % i)
        img = Image.open(im_name)
        img = img.resize((args['dim'][1], args['dim'][0]))

        im_list.append(np.array(img))

    # delete small annotation error
    for i in range(len(mask_list)):  # clean up artifacts in the annotation, must do
        for target in range(1, 50):
            if (mask_list[i] == target).sum() < 30:
                mask_list[i][mask_list[i] == target] = 0


    all_mask = np.stack(mask_list)


    # 3. Generate ALL training data for training pixel classifier
    all_feature_maps_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all), args['dim'][2]), dtype=np.float16)
    all_mask_train = np.zeros((args['dim'][0] * args['dim'][1] * len(latent_all),), dtype=np.float16)


    vis = []
    for i in range(len(latent_all) ):

        gc.collect()

        latent_input = latent_all[i].float()

        img, feature_maps = latent_to_image(g_all, upsamplers, latent_input.unsqueeze(0), dim=args['dim'][1],
                                            return_upsampled_layers=True, use_style_latents=args['annotation_data_from_w'])
        if args['dim'][0]  != args['dim'][1]:
            # only for car
            img = img[:, 64:448]
            feature_maps = feature_maps[:, :, 64:448]
        mask = all_mask[i:i + 1]
        feature_maps = feature_maps.permute(0, 2, 3, 1)

        feature_maps = feature_maps.reshape(-1, args['dim'][2])
        new_mask =  np.squeeze(mask)

        mask = mask.reshape(-1)

        all_feature_maps_train[args['dim'][0] * args['dim'][1] * i: args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = feature_maps.cpu().detach().numpy().astype(np.float16)
        all_mask_train[args['dim'][0] * args['dim'][1] * i:args['dim'][0] * args['dim'][1] * i + args['dim'][0] * args['dim'][1]] = mask.astype(np.float16)

        img_show =  cv2.resize(np.squeeze(img[0]), dsize=(args['dim'][1], args['dim'][1]), interpolation=cv2.INTER_NEAREST)

        curr_vis = np.concatenate( [im_list[i], img_show, colorize_mask(new_mask, palette)], 0 )

        vis.append( curr_vis )


    vis = np.concatenate(vis, 1)
    imageio.imsave(os.path.join(args['exp_dir'], "train_data.jpg"),
                      vis)

    return all_feature_maps_train, all_mask_train, num_data


def main(args
         ):

    all_feature_maps_train_all, all_mask_train_all, num_data = prepare_data(args, palette)


    train_data = trainData(torch.FloatTensor(all_feature_maps_train_all),
                           torch.FloatTensor(all_mask_train_all))


    count_dict = get_label_stas(train_data)

    max_label = max([*count_dict])
    print(" *********************** max_label " + str(max_label) + " ***********************")


    print(" *********************** Current number data " + str(num_data) + " ***********************")


    batch_size = args['batch_size']

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    print(" *********************** Current dataloader length " +  str(len(train_loader)) + " ***********************")

    for MODEL_NUMBER in range(args['model_num']):

        gc.collect()

        classifier = pixel_classifier(numpy_class=(max_label + 1), dim=args['dim'][-1])

        classifier.init_weights()

        classifier = nn.DataParallel(classifier, device_ids=device_ids).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        for epoch in range(100):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.type(torch.long)
                y_batch = y_batch.type(torch.long)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                loss = criterion(y_pred, y_batch)

                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % 1000 == 0:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item())
                    gc.collect()


                if iteration % 5000 == 0:
                    model_path = os.path.join(args['exp_dir'],
                                              'model_iter' +  str(iteration) + '_number_' + str(MODEL_NUMBER) + '.pth')
                    print('Save checkpoint, Epoch : ', str(epoch), ' Path: ', model_path)

                    torch.save({'model_state_dict': classifier.state_dict()},
                               model_path)

                if epoch > 3:
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > 50:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                        break

            if stop_sign == 1:
                break

        gc.collect()
        model_path = os.path.join(args['exp_dir'],
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
        gc.collect()


        gc.collect()
        torch.cuda.empty_cache()    # clear cache memory on GPU


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))
    main(opts)

