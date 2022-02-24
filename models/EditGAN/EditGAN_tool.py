# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
torch.manual_seed(0)
import json
import torch.nn.functional as F
import cv2
device_ids = [0]
from tqdm import tqdm
import scipy.misc
import timeit
from utils.data_utils import *
from utils.model_utils import *
import gc
from models.encoder.encoder import FPNEncoder
import argparse
import numpy as np
import os
import torch.optim as optim
from torchvision import transforms
import lpips as lpips
from utils.mask_manipulate_utils import *
import imageio

np.random.seed(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app, support_credentials=True)





class Tool(object):
    def __init__(self, ce_loss_weight=1, encoder_loss_weight=1):
        args_file = "experiments/tool_car.json"
        self.args = json.load(open(args_file, 'r'))

        resume = self.args['encoder_checkpoint']
        classfier_checkpoint =  self.args['classfier_checkpoint']
        self.root_path = self.args['root_path']


        self.editing_vector_path = os.path.join(self.root_path , "editing_vectors")
        self.sampling_path =  os.path.join(self.root_path ,"samples")
        self.result_path =  os.path.join(self.root_path ,"results")
        self.upload_latent_path =  os.path.join(self.root_path , "upload_latents")

        self.make_path()

        self.num_classifier = self.args['num_classifier']
        self.classifier_iter = self.args['classifier_iter']


        num_class =  self.args['num_class']
        self.use_noise = self.args['use_noise']
        self.g_all, self.upsamplers, self.bi_upsamplers, self.classifier_list, self.avg_latent = prepare_model(self.args, classfier_checkpoint,
                                                                             self.args['classifier_iter'],
                                                                             num_class, self.num_classifier)
        self.inter = Interpolate(self.args['im_size'][1], 'bilinear')


        self.stylegan_encoder = FPNEncoder(3, n_latent=self.args['n_latent'], only_last_layer=self.args['use_w'])
        self.stylegan_encoder = self.stylegan_encoder.to(device)

        self.stylegan_encoder.load_state_dict(torch.load(resume, map_location=device)['model_state_dict'], strict=True)
        self.steps = self.args['steps']
        self.embedding_steps = self.args['embedding_steps']

        self.rgb_loss_weight = self.args['rgb_loss_weight']
        self.ce_loss_weight = ce_loss_weight
        self.encoder_loss_weight = encoder_loss_weight

        self.percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True,
                                       normalize=self.args['normalize']).to(device)

        self.testing_latent_list = self.prepare_datasetGAN_data(self.args['datasetgan_testimage_embedding_path'])
        self.car_platte = car_32_platette_hex
        print("TOOL init!!")

    def make_path(self):

        if os.path.exists(self.root_path):
            pass
        else:
            os.system('mkdir -p %s' % (self.root_path))
            print('Experiment folder created at: %s' % (self.root_path))


        if os.path.exists(self.editing_vector_path):
            pass
        else:
            os.system('mkdir -p %s' % (self.editing_vector_path))
            print('Experiment folder created at: %s' % (self.editing_vector_path))

        if os.path.exists(self.sampling_path):
            pass
        else:
            os.system('mkdir -p %s' % (self.sampling_path))
            print('Experiment folder created at: %s' % (self.sampling_path))

        if os.path.exists(self.result_path):
            pass
        else:
            os.system('mkdir -p %s' % (self.result_path))
            print('Experiment folder created at: %s' % (self.result_path))

        if os.path.exists(self.upload_latent_path):
            pass
        else:
            os.system('mkdir -p %s' % (self.upload_latent_path))
            print('Experiment folder created at: %s' % (self.upload_latent_path))


    def prepare_datasetGAN_data(self, embedding_path):
        test_latent_list = []

        for i in tqdm(range(10)):
            curr_latent = np.load(os.path.join(embedding_path, 'latents_image_%0d.npy' % i))
            test_latent_list.append(curr_latent)
            optimized_latent = torch.from_numpy(curr_latent).type(torch.FloatTensor).to(device).unsqueeze(0)
            img_out, img_seg_final = self.run_seg(optimized_latent)

            imageio.imsave(os.path.join(self.root_path, 'images', 'car_real', str(i) + '.jpg'),
                              img_out[0, 64:448].astype(np.uint8))
            seg_vis = colorize_mask(img_seg_final, car_32_palette)
            imageio.imsave(os.path.join(self.root_path, 'images','car_real', 'colorize_mask',  str(i) + '.png'),
                              seg_vis)
        testing_latent_list = np.array(test_latent_list)
        return testing_latent_list


    def run_embedding(self, im):
        use_noise = self.use_noise
        label_im_tensor =  transforms.ToTensor()(im)

        label_im_tensor = label_im_tensor.unsqueeze(0).to(device)
        label_im_tensor = label_im_tensor * 2.0 - 1.0
        latent_in = self.stylegan_encoder(label_im_tensor)
        im_out_wo_encoder, _ = latent_to_image(self.g_all, self.upsamplers, latent_in,
                                               process_out=True, use_style_latents=True,
                                               return_only_im=True)

        out = run_embedding_optimization(self.args, self.g_all,
                                         self.bi_upsamplers, self.inter, self.percept,
                                         label_im_tensor, latent_in, steps=self.embedding_steps,
                                         stylegan_encoder=self.stylegan_encoder,
                                         use_noise=use_noise,
                                         noise_loss_weight=300
                                         )

        optimized_latent, optimized_noise, loss_cache = out

        img_out, img_seg_final = self.run_seg(optimized_latent)

        return img_out, img_seg_final, optimized_latent[0].detach().cpu().numpy(), optimized_noise

    # Testing time optimization
    def run_optimization_post_process(self, finetune_steps, latent_in, editing_vector, scale, editing_name, class_ids=[], noise=None):
        gc.collect()
        torch.cuda.empty_cache()  # clear cache memory on GPU
        start_time = timeit.default_timer()
        if class_ids == []:
            operation_name = editing_name.split("_")[0]
            class_ids = car_semantic_ids[operation_name]
        curr_latent = latent_in + editing_vector * scale
        curr_latent.requires_grad = True
        optimized_latent = curr_latent
        if finetune_steps > 1:
            optimizer = optim.Adam([curr_latent], lr=1e-6)
            with torch.no_grad():
                img_out, affine_layers = latent_to_image(self.g_all, self.upsamplers, latent_in, process_out=False,
                                                         return_upsampled_layers=False,
                                                         use_style_latents=True, return_only_im=False)
                img_out = self.inter(img_out)
                img_tensor = (img_out + 1.0) / 2.0
                image_features = []
                for i in range(len(affine_layers)):
                    image_features.append(self.upsamplers[i](
                        affine_layers[i]))
                image_features = torch.cat(image_features, 1)
                image_features = image_features[0]
                image_features = image_features.reshape(self.args['dim'], -1).transpose(1, 0)
                seg_mode_ensemble = []
                for MODEL_NUMBER in range(self.num_classifier):
                    classifier = self.classifier_list[MODEL_NUMBER]
                    img_seg = classifier(image_features)
                    seg_mode_ensemble.append(img_seg.unsqueeze(0))
                mask_before_edit = torch.argmax(torch.mean(torch.cat(seg_mode_ensemble, 0), 0), 1).reshape(512,
                                                                                                        512).detach().cpu().numpy()
                _, affine_layers = latent_to_image(self.g_all, self.upsamplers, curr_latent, process_out=False,
                                                         return_upsampled_layers=False,
                                                         use_style_latents=True, return_only_im=False)

                image_features = []
                for i in range(len(affine_layers)):
                    image_features.append(self.upsamplers[i](
                        affine_layers[i]))
                image_features = torch.cat(image_features, 1)
                image_features = image_features[0]
                image_features = image_features.reshape(self.args['dim'], -1).transpose(1, 0)
                seg_mode_ensemble = []
                for MODEL_NUMBER in range(self.num_classifier):
                    classifier = self.classifier_list[MODEL_NUMBER]
                    img_seg = classifier(image_features)
                    seg_mode_ensemble.append(img_seg.unsqueeze(0))
                org_mask = torch.argmax(torch.mean(torch.cat(seg_mode_ensemble, 0), 0), 1).reshape(512,
                                                                                                        512).detach().cpu().numpy()
                roi = np.zeros((512, 512), np.uint8)
                for ids in class_ids:
                    roi += (org_mask == ids).astype(np.uint8)
                for ids in class_ids:
                    roi += (mask_before_edit == ids).astype(np.uint8)
                roi = (roi > 0)
            best_loss = 1e10

            del (seg_mode_ensemble)
            gc.collect()
            torch.cuda.empty_cache()
            kernel = np.ones((3, 3), np.uint8)
            dilate_roi = cv2.dilate(np.float32(roi), kernel, iterations=3).astype(np.uint8)

            dilate_roi_mask = 1 - torch.from_numpy(dilate_roi).unsqueeze(0).unsqueeze(0).cuda()
            ROI_mask = torch.from_numpy(org_mask[dilate_roi > 0]).unsqueeze(0).cuda()

            all_loss = []
            ce_criterion = nn.CrossEntropyLoss()

            loss_dict = {'p_loss': [], 'mse_loss': [], 'encoder_loss': [], 'ce_loss': [], 'error_ce_loss': [],
                         'g_loss': []}
            for _ in range(1, finetune_steps):
                loss = 0
                lr = 0.02
                optimizer.param_groups[0]['lr'] = lr
                img_out, affine_layers = latent_to_image(self.g_all, self.upsamplers, curr_latent, process_out=False,
                                                         return_upsampled_layers=False,
                                                         use_style_latents=True, return_only_im=False, noise=noise)
                image_features = []
                for i in range(len(affine_layers)):
                    curr_up_feature = self.upsamplers[i](
                        affine_layers[i])
                    image_features.append(curr_up_feature)
                image_features = torch.cat(image_features, 1)
                img_out = self.inter(img_out)
                img_out = (img_out + 1.0) / 2.0
                p_loss = self.percept(img_out * dilate_roi_mask, img_tensor * dilate_roi_mask).mean()
                mse_loss = F.mse_loss(img_out * dilate_roi_mask, img_tensor * dilate_roi_mask, reduction='none')
                loss_dict['p_loss'].append(p_loss.item())
                loss_dict['mse_loss'].append(mse_loss.mean().item())
                loss += self.rgb_loss_weight * (self.args['loss_dict']['p_loss'] * p_loss + \
                                           5 * self.args['loss_dict']['mse_loss'] * mse_loss.mean())
                record_loss = self.args['loss_dict']['p_loss'] * p_loss + self.args['loss_dict']['mse_loss'] * mse_loss.mean()
                roi_features = image_features[:, :, dilate_roi]
                roi_features = roi_features[0]
                roi_features = roi_features.reshape(self.args['dim'], -1).transpose(1, 0)
                # 512 * 512 * 6016
                seg_mode_ensemble = []
                for MODEL_NUMBER in range(self.num_classifier):
                    classifier = self.classifier_list[MODEL_NUMBER]
                    img_seg = classifier(roi_features)
                    seg_mode_ensemble.append(img_seg.unsqueeze(0))

                seg_mode_ensemble = torch.mean(torch.cat(seg_mode_ensemble, 0), 0)
                ce_loss = ce_criterion(seg_mode_ensemble, ROI_mask[0])
                loss_dict['ce_loss'].append(ce_loss.item())
                loss += ce_loss * self.ce_loss_weight
                optimizer.zero_grad()
                loss.backward()
                all_loss.append(record_loss.item())
                optimizer.step()

                del (image_features, roi_features)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    count = 0
                    optimized_latent = curr_latent.detach()

        img_out, img_seg_final = self.run_seg(optimized_latent)
        print("apply_editing_vector time,", timeit.default_timer() - start_time)
        gc.collect()
        torch.cuda.empty_cache()  # clear cache memory on GPU
        return img_out, img_seg_final, optimized_latent[0].detach().cpu().numpy()

    def run_seg(self, optimized_latent):
        img_out, affine_layers = latent_to_image(self.g_all, self.bi_upsamplers, optimized_latent, process_out=True,
                                                 return_upsampled_layers=False,
                                                 use_style_latents=True, return_only_im=False)
        image_features = []
        for i in range(len(affine_layers)):
            image_features.append(self.bi_upsamplers[i](
                affine_layers[i]))
        image_features = torch.cat(image_features, 1)
        image_features = image_features[:, :, 64:448]
        image_features = image_features[0]
        image_features = image_features.reshape(self.args['dim'], -1).transpose(1, 0)
        seg_mode_ensemble = []
        for MODEL_NUMBER in range(self.num_classifier):
            classifier = self.classifier_list[MODEL_NUMBER]
            img_seg = classifier(image_features)
            seg_mode_ensemble.append(img_seg.unsqueeze(0))
        img_seg_final = torch.argmax(torch.mean(torch.cat(seg_mode_ensemble, 0), 0),1).reshape(384, 512).detach().cpu().numpy()
        del (affine_layers)
        return img_out, img_seg_final

    def run_optimization_editGAN(self, org_mask, latent_in, roi, noise=None):
        gc.collect()
        torch.cuda.empty_cache()
        kernel = np.ones((3, 3), np.uint8)
        dilate_roi = cv2.dilate(np.float32(roi), kernel, iterations=3).astype(np.uint8)
        dilate_roi_mask = 1 - torch.from_numpy(dilate_roi).unsqueeze(0).unsqueeze(0).cuda()
        ROI_mask = torch.from_numpy(org_mask[dilate_roi > 0]).unsqueeze(0).cuda()
        org_latnet_in = copy.deepcopy(latent_in)
        if self.args['truncation']:
            latent_in = self.g_all.module.truncation(latent_in)
        latent_in.requires_grad = True
        optimizer = optim.Adam([latent_in], lr=1e-6)
        best_loss = 1e10
        count = 0
        with torch.no_grad():
            img_out, _ = latent_to_image(self.g_all, self.upsamplers, org_latnet_in,
                                         process_out=False, use_style_latents=True, noise=noise,
                                         return_only_im=True)
            img_out = self.inter(img_out)
            img_tensor = (img_out + 1.0) / 2.0

        optimized_latent = latent_in
        all_loss = []
        ce_criterion = nn.CrossEntropyLoss()
        anneal_count = 1
        for _ in tqdm(range(1, self.steps)):
            loss = 0
            lr = 0.02
            optimizer.param_groups[0]['lr'] = lr
            if _ % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()  # clear cache memory on GPU
            img_out, affine_layers = latent_to_image(self.g_all, self.upsamplers, latent_in, process_out=False,
                                                     return_upsampled_layers=False, noise=noise,
                                                     use_style_latents=True, return_only_im=False)
            image_features = []
            for i in range(len(affine_layers)):
                image_features.append(self.upsamplers[i](
                    affine_layers[i]))
            image_features = torch.cat(image_features, 1)
            img_out = self.inter(img_out)
            img_out = (img_out + 1.0) / 2.0
            p_loss = self.percept(img_out * dilate_roi_mask, img_tensor * dilate_roi_mask).mean()
            mse_loss = F.mse_loss(img_out * dilate_roi_mask, img_tensor * dilate_roi_mask, reduction='none')
            encoder_loss = F.mse_loss(latent_in, self.stylegan_encoder(img_out).detach())
            loss += self.rgb_loss_weight * (self.args['loss_dict']['p_loss'] * p_loss + \
                                       5 * self.args['loss_dict'][
                                           'mse_loss'] * mse_loss.mean()) + self.encoder_loss_weight * encoder_loss

            record_loss = self.args['loss_dict']['p_loss'] * p_loss + self.args['loss_dict']['mse_loss'] * mse_loss.mean()
            roi_features = image_features[:, :, dilate_roi]
            roi_features = roi_features[0]
            roi_features = roi_features.reshape(self.args['dim'], -1).transpose(1, 0)
            seg_mode_ensemble = []
            for MODEL_NUMBER in range(self.num_classifier):
                classifier = self.classifier_list[MODEL_NUMBER]
                img_seg = classifier(roi_features)
                seg_mode_ensemble.append(img_seg.unsqueeze(0))
            seg_mode_ensemble = torch.mean(torch.cat(seg_mode_ensemble, 0), 0)
            ce_loss = ce_criterion(seg_mode_ensemble, ROI_mask[0].long())
            loss += ce_loss * self.ce_loss_weight
            optimizer.zero_grad()
            loss.backward()
            all_loss.append(record_loss.item())
            optimizer.step()
            del (image_features, roi_features)
            if loss.item() < best_loss:
                best_loss = loss.item()
                count = 0
                optimized_latent = latent_in.detach()
            else:
                count += 1
        gc.collect()
        torch.cuda.empty_cache()
        img_out, img_seg_final = self.run_seg(optimized_latent)
        return img_out, img_seg_final, optimized_latent[0].detach().cpu().numpy()

    def run_sampling(self):
        with torch.no_grad():
            latent = np.random.randn(1, 512)
            latent_in = torch.from_numpy(latent).type(torch.FloatTensor).to(device)

            style_latents = latent_to_image(self.g_all, self.bi_upsamplers, latent_in, return_stylegan_latent=True)
            img_out, img_seg_final = self.run_seg(style_latents)

        img_out = img_out[0, 64:448]
        return img_out, img_seg_final, style_latents[0].detach().cpu().numpy()



