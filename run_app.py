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
import flask
import imageio
torch.manual_seed(0)
import json
import pickle
from flask import Blueprint, render_template
import os
import cv2

device_ids = [0]
from PIL import Image
import timeit
from utils.poisson_image_editing import poisson_edit
from utils.data_utils import *
from utils.model_utils import *
import numpy as np
import argparse
import copy
from io import BytesIO
from models.EditGAN.EditGAN_tool import Tool

np.random.seed(6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app, support_credentials=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8888)

    args = parser.parse_args()

    return args


@app.route('/')
def index():
    global tool
    tool = Tool()

    return render_template('index.html')


@app.route('/api/edit_from_mask', methods=['POST'])
@cross_origin(supports_credentials=True)
def edit_from_mask():
    data = request.get_json(force=True)

    # load mask
    base64im = data['imageBase64']
    # extension = base64im.split('/')[1].split(';')[0]
    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(base64im.encode())))

    seg = np.asarray(im)[:, :, :-1]

    seg_mask = np.zeros((seg.shape[0], seg.shape[1]))

    for i in range(int(len(car_32_palette) / 3)):
        curr_palette = [car_32_palette[3 * i], car_32_palette[3 * i + 1], car_32_palette[3 * i + 2]]
        id = np.all(seg == np.array(curr_palette), 2)
        seg_mask[id == 1] = i
    if seg_mask.shape[0] != seg_mask.shape[1]:
        canvas = np.zeros([seg_mask.shape[1], seg_mask.shape[1]], dtype=np.uint8)
        canvas[(seg_mask.shape[1] - seg_mask.shape[0]) // 2: (seg_mask.shape[1] + seg_mask.shape[0]) // 2, :] = seg_mask

        seg_mask = canvas

    # load roi
    base64im = data['roi']
    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(base64im.encode())))
    roi = (np.asarray(im)[:, :, 0]) == 0
    if roi.shape[0] != roi.shape[1]:
        canvas = np.zeros([roi.shape[1], roi.shape[1]], dtype=np.uint8)
        canvas[(roi.shape[1] - roi.shape[0]) // 2: (roi.shape[1] + roi.shape[0]) // 2, :] = roi

        roi = canvas
    if data['image_id'][:8] == "results_":

        curr_latent = np.load(os.path.join(tool.result_path, data['image_id'][8:] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)

    elif data['image_id'][:7] == "sample_":

        curr_latent = np.load(os.path.join(tool.sampling_path, data['image_id'] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
    elif data['image_id'][:7] == "upload_":
        curr_latent = torch.from_numpy(
            np.load(os.path.join(tool.upload_latent_path, data['image_id'][7:] + '_latent.npy'))).cuda().unsqueeze(0)
    else:
        curr_image_id = int(data['image_id'])
        print("Current image id: ", curr_image_id)
        curr_latent = torch.from_numpy(tool.testing_latent_list[curr_image_id]).cuda().unsqueeze(0)
    org_latent = copy.deepcopy(curr_latent)

    img_out, img_seg_final, optimized_latent = tool.run_optimization_editGAN(seg_mask, curr_latent, roi)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_latent.npy'), optimized_latent)

    roi_colors = data['roi_id']
    roi_color_color_ids = []
    for color in roi_colors:
        roi_color_color_ids.append(tool.car_platte.index(color))

    dump_dict = {"edit_vector": optimized_latent - org_latent.detach().squeeze(0).cpu().numpy(),
                 "roi_ids": roi_color_color_ids}

    with open(os.path.join(tool.result_path, 'current_editing_latent_cache.pickle'), 'wb') as handle:
        pickle.dump(dump_dict, handle)

    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_mask_org.npy'), img_seg_final.astype(np.uint8))

    sv_name = os.path.join(tool.result_path, data['image_id'] + ".jpg")
    imageio.imsave(sv_name, img_out[0, 64:448].astype(np.uint8))

    return flask.make_response(json.dumps({"sv_name": sv_name.split("/")[-1].split(".")[0]}), 200)


@app.route('/api/apply_current_editing_vector', methods=['POST'])
@cross_origin(supports_credentials=True)
def apply_current_editing_vector():
    data = request.get_json(force=True)

    # load roi
    base64im = data['roi']
    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    im = Image.open(BytesIO(base64.b64decode(base64im.encode())))
    roi = (np.asarray(im)[:, :, 0]) == 0
    if roi.shape[0] != roi.shape[1]:
        canvas = np.zeros([roi.shape[1], roi.shape[1]], dtype=np.uint8)
        canvas[(roi.shape[1] - roi.shape[0]) // 2: (roi.shape[1] + roi.shape[0]) // 2, :] = roi
        roi = canvas

    with open(os.path.join(tool.result_path, 'current_editing_latent_cache.pickle'), 'rb') as handle:
        dump_dict = pickle.load(handle)

    if data['image_id'][:8] == "results_":

        curr_latent = np.load(os.path.join(tool.result_path, data['image_id'][8:] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
    elif data['image_id'][:7] == "sample_":

        curr_latent = np.load(os.path.join(tool.sampling_path, data['image_id'] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
    elif data['image_id'][:7] == "upload_":
        curr_latent = torch.from_numpy(
            np.load(os.path.join(tool.upload_latent_path, data['image_id'][7:] + '_latent.npy'))).cuda().unsqueeze(0)

    else:

        curr_image_id = int(data['image_id'].split('_')[-1])
        print("Current image id: ", curr_image_id)

        curr_latent = torch.from_numpy(tool.testing_latent_list[curr_image_id]).cuda().unsqueeze(0)

    editing_vector = dump_dict['edit_vector']
    editing_vector = torch.from_numpy(editing_vector).cuda()

    scale = float(data['scale']) / 2.
    finetune_steps = int(data['steps'])

    img_out, img_seg_final, optimized_latent = tool.run_optimization_post_process(finetune_steps, curr_latent,
                                                                                  editing_vector, scale, "",
                                                                                  class_ids=dump_dict['roi_ids'])

    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '.jpg'),
                      img_out[0, 64:448].astype(np.uint8))
    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_mask_org.npy'), img_seg_final.astype(np.uint8))

    np.save(os.path.join(tool.result_path, data['image_id'] + '_latent.npy'), optimized_latent)

    return flask.make_response(json.dumps({"sv_name": str(data['image_id'])}), 200)


@app.route('/api/random_roll', methods=['POST'])
@cross_origin(supports_credentials=True)
def random_roll():
    start_time = timeit.default_timer()
    img_out, img_seg_final, latent = tool.run_sampling()
    print("run_sampling time,", timeit.default_timer() - start_time)

    random_im_id = np.random.randint(10000, size=1)[0]

    imageio.imsave(os.path.join(tool.sampling_path, "sample_" + str(random_im_id) + '.jpg'), img_out)
    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.sampling_path, "sample_" + str(random_im_id) + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.sampling_path, "sample_" + str(random_im_id) + '_latent.npy'), latent)

    return flask.make_response(json.dumps({"sv_name": str(random_im_id)}), 200)


@app.route('/api/apply_editing_vector', methods=['POST'])
@cross_origin(supports_credentials=True)
def apply_editing_vector():
    data = request.get_json(force=True)
    editing_vector_name = data['editing_vector_id']

    editing_vector = np.load(os.path.join(tool.editing_vector_path, editing_vector_name + '.npy'))
    editing_vector = torch.from_numpy(editing_vector).cuda()

    scale = float(data['scale']) / 2.
    finetune_steps = int(data['steps'])
    if data['image_id'][:8] == "results_":

        curr_latent = np.load(os.path.join(tool.result_path, data['image_id'][8:] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
    elif data['image_id'][:7] == "sample_":

        curr_latent = np.load(os.path.join(tool.sampling_path, data['image_id'] + '_latent.npy'))
        curr_latent = torch.from_numpy(curr_latent).cuda().unsqueeze(0)
    elif data['image_id'][:7] == "upload_":
        curr_latent = torch.from_numpy(
            np.load(os.path.join(tool.upload_latent_path, data['image_id'][7:] + '_latent.npy'))).cuda().unsqueeze(0)
    else:
        curr_image_id = int(data['image_id'])
        print("Current image id: ", curr_image_id)
        curr_latent = torch.from_numpy(tool.testing_latent_list[curr_image_id]).cuda().unsqueeze(0)

    img_out, img_seg_final, optimized_latent = tool.run_optimization_post_process(finetune_steps, curr_latent,
                                                                                  editing_vector, scale,
                                                                                  editing_vector_name)

    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '.jpg'),
                      img_out[0, 64:448].astype(np.uint8))
    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_mask_org.npy'), img_seg_final.astype(np.uint8))

    np.save(os.path.join(tool.result_path, data['image_id'] + '_latent.npy'), optimized_latent)

    return flask.make_response(json.dumps({"sv_name": str(data['image_id'])}), 200)


@app.route('/upload_crop_image', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_crop():
    data = request.get_json(force=True)
    # load mask
    base64im = data['imageBase64']
    # extension = base64im.split('/')[1].split(';')[0]
    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(base64im.encode()))).convert('RGB')

    img = img.resize((512, 384))
    img = np.asarray(img)

    canvas = np.zeros([512, 512, 3], dtype=np.uint8)
    canvas[(512 - 384) // 2: (512 + 384) // 2, :, :] = img

    canvas = Image.fromarray(canvas, 'RGB')

    img_out, img_seg_final, optimized_latent, optimized_noise = tool.run_embedding(canvas)

    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '.jpg'),
                      img_out[0, 64:448].astype(np.uint8))
    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_latent.npy'), optimized_latent)
    np.save(os.path.join(tool.upload_latent_path, data['image_id'] + '_latent.npy'), optimized_latent)

    return flask.make_response(json.dumps({"sv_name": str(data['image_id'])}), 200)


@app.route('/upload_image', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload():
    data = request.get_json(force=True)
    # load mask
    base64im = data['imageBase64']
    # extension = base64im.split('/')[1].split(';')[0]
    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(base64im.encode()))).convert('RGB')
    org_img = copy.deepcopy(img)
    crop_img, bbox_valid = crop_from_bbox(np.asarray(img), data['crop_loc'])

    data['bbox_final'] = bbox_valid
    crop_img = Image.fromarray(crop_img)

    crop_img = crop_img.resize((512, 384))
    crop_img = np.asarray(crop_img)

    canvas = np.zeros([512, 512, 3], dtype=np.uint8)
    canvas[(512 - 384) // 2: (512 + 384) // 2, :, :] = crop_img

    canvas = Image.fromarray(canvas, 'RGB')

    img_out, img_seg_final, optimized_latent, optimized_noise = tool.run_embedding(canvas)

    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_wo_crop.jpg'),
                      np.asarray(org_img).astype(np.uint8))

    with open(os.path.join(tool.result_path, data['image_id'] + '.json'), 'w') as f:
        json.dump(data, f)

    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '.jpg'),
                      img_out[0, 64:448].astype(np.uint8))

    seg_vis = colorize_mask(img_seg_final, car_32_palette)
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_mask.png'),
                      seg_vis)

    np.save(os.path.join(tool.result_path, data['image_id'] + '_mask_org.npy'), img_seg_final.astype(np.uint8))

    np.save(os.path.join(tool.result_path, data['image_id'] + '_latent.npy'), optimized_latent)
    np.save(os.path.join(tool.upload_latent_path, data['image_id'] + '_latent.npy'), optimized_latent)

    return flask.make_response(json.dumps({"sv_name": str(data['image_id'])}), 200)


@app.route('/download_image', methods=['POST'])
@cross_origin(supports_credentials=True)
def download_image():
    data = request.get_json(force=True)
    # load mask
    base64im = data['imageBase64']

    t = base64im.split('/')[0].split(':')[1]
    assert t == 'image', 'Did not get image data!'
    base64im = base64im.split(',')[1]
    img = Image.open(BytesIO(base64.b64decode(base64im.encode()))).convert('RGB')
    if data['image_id'][:8] == "results_":
        mask_image_id = copy.deepcopy(data['image_id'])[8:]

    image_id = data['image_id']

    while image_id[:8] == "results_":
        image_id = image_id[8:]

    if image_id[:7] == "upload_":
        image_id = image_id[7:]

    with open(os.path.join(tool.result_path, image_id + '.json'), 'r') as f:
        data = json.load(f)

    bbox = data['bbox_final']

    img_org = np.asarray(Image.open(os.path.join(tool.result_path, image_id + '_wo_crop.jpg'))).astype(np.uint8)

    img_full = crop2fullImg(np.asarray(img), bbox, img_org * 0., im_size=img_org.shape)

    img_mask = np.load(os.path.join(tool.result_path, mask_image_id + '_mask_org.npy'))
    img_mask_final = crop2fullImg(img_mask, bbox, (img_org * 0)[:, :, 0], im_size=img_org.shape)

    img_mask_final = cv2.dilate(np.float32(img_mask_final > 0), np.ones((3, 3), np.uint8), iterations=3).astype(
        np.uint8)

    img_final = poisson_edit(img_full, img_org, img_mask_final, [0, 0])
    imageio.imsave(os.path.join(tool.result_path, data['image_id'] + '_final.jpg'),
                      img_final.astype(np.uint8))

    return flask.make_response(json.dumps({"sv_name": str(data['image_id'])}), 200)


@app.route('/upload_vector', methods=['POST'])
@cross_origin(supports_credentials=True)
def upload_vector():
    f = request.files['file']

    f.save(os.path.join(tool.result_path, 'current_editing_latent_cache.pickle'))
    return 'file uploaded successfully'


if __name__ == '__main__':
    args = get_args()

    app.run(host='0.0.0.0', threaded=True, port=args.port)
