# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import numpy as np
import copy
import cv2
import PIL

def mask_to_bbox(mask):
    mask = (mask > 0)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item(), rmax.item()]  # xywh


palette = [1.0000, 1.0000, 1.0000,
           0.4420, 0.5100, 0.4234,
           0.8562, 0.9537, 0.3188,
           0.2405, 0.4699, 0.9918,
           0.8434, 0.9329, 0.7544,
           0.3748, 0.7917, 0.3256,
           0.0190, 0.4943, 0.3782,
           0.7461, 0.0137, 0.5684,
           0.1644, 0.2402, 0.7324,
           0.0200, 0.4379, 0.4100,
           0.5853, 0.8880, 0.6137,
           0.7991, 0.9132, 0.9720,
           0.6816, 0.6237, 0.8562,
           0.9981, 0.4692, 0.3849,
           0.5351, 0.8242, 0.2731,
           0.1747, 0.3626, 0.8345,
           0.5323, 0.6668, 0.4922,
           0.2122, 0.3483, 0.4707,
           0.6844, 0.1238, 0.1452,
           0.3882, 0.4664, 0.1003,
           0.2296, 0.0401, 0.3030,
           0.5751, 0.5467, 0.9835,
           0.1308, 0.9628, 0.0777,
           0.2849, 0.1846, 0.2625,
           0.9764, 0.9420, 0.6628,
           0.3893, 0.4456, 0.6433,
           0.8705, 0.3957, 0.0963,
           0.6117, 0.9702, 0.0247,
           0.3668, 0.6694, 0.3117,
           0.6451, 0.7302, 0.9542,
           0.6171, 0.1097, 0.9053,
           0.3377, 0.4950, 0.7284,
           0.1655, 0.9254, 0.6557,
           0.9450, 0.6721, 0.6162]

palette = [int(item * 255) for item in palette]


def color_mask_to_seg(mask):
    seg_mask = np.zeros((mask.shape[0], mask.shape[1]))
    print(seg_mask.shape)

    rgb_to_id_dict = {}
    for i in range(int(len(palette) / 3)):
        color = palette[3 * i: 3 * i + 3]

        ids1 = np.all(mask == np.array(color), 2)
        seg_mask[ids1 == 1] = i

    return seg_mask


################################ Bird ################################

bird_semantic_ids = {"beak": [3, 10], "eyes": [11], "tail": [18, 16], "wing": [20], "head": [3, 10, 9, 11, 6],
                     "belly": [5, 18, 16]}


def delete_tail(source_mask):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["tail"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 0
    return new_mask, delete


def belly_enlarge(source_mask, scale):
    ids = bird_semantic_ids['belly']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    A = np.float32([[1, 0, 0], [0, 1, 20]])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def tail_large(source_mask, scale):
    ids = bird_semantic_ids['tail']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)
    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def wing_enlarge(source_mask, scale):
    ids = bird_semantic_ids['wing']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    A = np.float32([[1, 0, -30], [0, 1, 30]])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def wing_rotate(source_mask):
    ids = bird_semantic_ids['wing']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)

    new_mask[roi] = 1

    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D((bbox[2], bbox[0]), -20, 1)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    A = np.float32([[1, 0, -10], [0, 1, 30]])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def head_rotate(source_mask):
    ids = bird_semantic_ids['head']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 30, 1)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def delete_beak(source_mask):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["beak"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 0
    return new_mask, delete


def wide_beak_12(source_mask, factor):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["beak"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 9

    target_mask = copy.deepcopy(source_mask)
    target_mask[delete == 0] = 0

    target_mask_res = cv2.resize(target_mask, (int(factor * w), h), interpolation=cv2.INTER_NEAREST)

    target_mask_res = target_mask_res[:, -512:]

    A = np.float32([[1, 0, 100], [0, 1, 0]])
    target_mask_res = cv2.warpAffine(target_mask_res.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        roi += (target_mask_res == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(target_mask_res == id)] = id

    return new_mask, roi


def delete_tail(source_mask):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["tail"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 0
    return new_mask, delete


def delete_wing(source_mask):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["wing"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 0
    return new_mask, delete


def wide_beak(source_mask, factor):
    h, w = source_mask.shape[:2]

    roi = np.zeros((source_mask.shape[0], source_mask.shape[1]))

    new_mask = copy.deepcopy(source_mask)
    ids = bird_semantic_ids["beak"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 0

    target_mask = copy.deepcopy(source_mask)
    target_mask[delete == 0] = 0

    target_mask_res = cv2.resize(target_mask, (int(factor * w), h), interpolation=cv2.INTER_NEAREST)

    target_mask_res = target_mask_res[:, -512:]

    A = np.float32([[1, 0, 200], [0, 1, 0]])
    target_mask_res = cv2.warpAffine(target_mask_res.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        roi += (target_mask_res == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(target_mask_res == id)] = id

    return new_mask, roi


def bird_enlarge_beak(source_mask, scale):
    ids = bird_semantic_ids['beak']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def bird_enlarge_eye(source_mask, scale):
    ids = bird_semantic_ids['eyes']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    roi += (ref_mask == ids[0])

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[ref_mask == ids[0]] = ids[0]

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


################################ Bedroom ################################
#
#  ['background', 'bed', 'bed***footboard', 'bed***headboard', 'bed***side rail',
# 'carpet', 'ceiling', 'ceiling fan***blade', 'curtain', 'cushion', 'floor',
# 'night table', 'night table***top', 'picture', 'pillow', 'table lamp***column', '
# table lamp***shade', 'wall', 'pane']
#

bedroom_semantic_ids = {"pillow": [14], "picture": [13]}


def add_picture(source_mask, target_mask):
    ids = bedroom_semantic_ids['picture']

    h, w = source_mask.shape[:2]

    new_mask = copy.deepcopy(source_mask)

    ref_mask = copy.deepcopy(target_mask)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)
    ref_mask = ref_mask * roi

    for id in ids:
        new_mask[(ref_mask == id)] = id

    for id in ids:
        roi += (new_mask == id)

    return new_mask, (roi > 0)


def delete_picture(source_mask):
    ids = bedroom_semantic_ids['picture']

    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 17
    all_roi += roi

    return new_mask, (all_roi > 0)


def delete_pillow(source_mask):
    ids = bedroom_semantic_ids['pillow']

    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 1
    all_roi += roi

    return new_mask, (all_roi > 0)


################################ Car ################################
# ['background', 'back bumper', 'bumper', 'car body', 'car_light_right', 'car_light_left',
# 'door_back', 'fender','door_front', 'grilles', 'back handle', 'fronthandle', 'hoods', 'license_plate_front',
# 'licence_plate_back','logo','mirror','roof','running boards', 'taillight right',
# 'taillight left','back wheel', 'front wheel','trunks','wheelhub_back','wheelhub_front','spoke_back',
# 'spoke_front', 'door_window_back', 'back windshield', 'door_window_front', 'windshield'

car_semantic_ids = {"frontlight": [4, 5], "wheel": [21, 22, 24, 25, 26, 27], "frontwheel": [22, 25, 27],
                    "handle": [10, 11],
                    "mirror": [16], "licenseplate": [13, 14], "spoke": [26, 27],
                    "window": [16, 17, 30, 28], "Sampling": [16, 17, 30, 28], "backwindow": [28],
                    "carback": [28, 29, 17]}


def add_back_window(source_mask):
    ids = car_semantic_ids['backwindow']
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    ref_mask = ref_mask * roi

    bbox = mask_to_bbox(roi)

    # A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, 2)
    # ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    ref_mask = cv2.resize(ref_mask, (int(5 * w), h), interpolation=cv2.INTER_NEAREST)

    ref_mask = ref_mask[:, -512:]

    A = np.float32([[1, 0, 100], [0, 1, -20]])

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[(ref_mask == id)] = id
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    for id in ids:
        all_roi += (new_mask == id)

    return new_mask, (all_roi > 0)


def delete_backwindshield(source_mask):
    ids = car_semantic_ids['backwindshield']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.dilate(np.float32(roi), kernel, iterations=3).astype(np.uint8)
    roi = (roi > 0)
    new_mask[roi] = 0
    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)
    # new_mask[ (ref_mask ==16)] = 16
    new_mask[(ref_mask == 31)] = 31

    return new_mask, (all_roi > 0)


def delete_sidewindow(source_mask):
    ids = car_semantic_ids['window']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.dilate(np.float32(roi), kernel, iterations=2).astype(np.uint8)
    roi = (roi > 0)
    new_mask[roi] = 0
    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)
    # new_mask[ (ref_mask ==16)] = 16
    new_mask[(ref_mask == 31)] = 31

    return new_mask, (all_roi > 0)


def rotate_spoke(source_mask):
    spoke_ids = car_semantic_ids['spoke']
    ids = [27]
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 25

    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), -50, 1)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)
    for id in ids:
        new_mask[ref_mask == id] = id
    all_roi += roi

    ids = [26]
    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 24

    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), -30, 1)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)
    for id in ids:
        new_mask[ref_mask == id] = id
    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in spoke_ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def delete_licnse_plate(source_mask):
    ids = car_semantic_ids['licenseplate']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 2
    all_roi += roi

    return new_mask, (all_roi > 0)


def delete_mirror(source_mask):
    ids = car_semantic_ids['mirror']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 30
    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_mirror(source_mask, scale):
    ids = car_semantic_ids['mirror']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    roi += (ref_mask == ids[0])
    new_mask[roi > 0] = 30
    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[ref_mask == ids[0]] = ids[0]

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def delete_handle(source_mask):
    ids = car_semantic_ids['handle']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, 400:] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)
    roi = (roi > 0)
    new_mask[roi] = 6
    all_roi += roi
    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :400] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 8

    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_handle(source_mask, scale):
    ids = car_semantic_ids['handle']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    roi += (ref_mask == ids[0])

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[ref_mask == ids[0]] = ids[0]

    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    roi += (ref_mask == ids[1])

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[ref_mask == ids[1]] = ids[1]

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_frontlight(source_mask, scale):
    ids = car_semantic_ids['frontlight']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, 100:] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :100] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_frontwheel(source_mask, scale):
    ids = car_semantic_ids['frontwheel']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, 300:] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :300] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_wheel(source_mask, scale):
    ids = car_semantic_ids['wheel']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, 300:] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :300] = 1

    ref_mask = copy.deepcopy(source_mask)
    ref_mask = ref_mask * half_mask

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (ref_mask == id)

    roi = (roi > 0)
    new_mask[roi] = 0
    ref_mask = ref_mask * roi
    bbox = mask_to_bbox(roi)
    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[ref_mask == id] = id

    all_roi += roi

    roi = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))
    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def change_front_light_shape(source_mask):
    seg_rgb = np.array(PIL.Image.open("./data/edit_images/8.png").convert('RGB'))
    new_mask = color_mask_to_seg(seg_rgb).astype(np.long)

    roi = source_mask - new_mask
    roi = (roi != 0) + (new_mask == 4) + (new_mask == 5)

    return new_mask, (roi > 0)


################################ Cat ################################

# ['background',
#  'cat',
#     'back',
#     'belly',
#     'chest',
#     'leg',
#     'paw',
#     'head',
#     'ear',
#     'eye',
#     'mouth',
#     'tongue',
#     'nose',
#     'tail',
#     'whiskers']


cat_semantic_ids = {"ear": [8], "eyes": [9], "eye": [9], "nose": [12], "mouth": [10, 11]}


def smaller_eyes(source_mask, scale):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, int(new_mask.shape[1] / 2):] = 1

    ref_mask = ref_mask * half_mask
    roi = (ref_mask == 8)
    roi = (roi > 0)

    new_mask[roi > 0] = 0

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 8)] = 8

    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :int(new_mask.shape[1] / 2)] = 1

    ref_mask = ref_mask * half_mask
    roi = (ref_mask == 9)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 9)] = 9

    roi = (new_mask == 9)

    all_roi += roi

    all_roi = (all_roi > 0)
    roi = all_roi

    return new_mask.astype(np.long), (roi > 0)


def move_cat_eyes(source_mask):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, int(new_mask.shape[0] / 2):] = 1

    ref_mask = copy.deepcopy(source_mask) * half_mask

    roi = (ref_mask == 9)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    ref_mask = ref_mask * roi

    all_roi += roi
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, (bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 9)] = 9

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :int(new_mask.shape[0] / 2)] = 1

    ref_mask = copy.deepcopy(source_mask) * half_mask

    roi = (ref_mask == 9)
    roi = (roi > 0)
    new_mask[roi > 0] = 7

    ref_mask = ref_mask * roi

    all_roi += roi

    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, -(bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 9)] = 9

    all_roi += (new_mask == 9)

    return new_mask, (all_roi > 0)


def delete_cat_ear(source_mask):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = (ref_mask == 8)
    roi = (roi > 0)
    new_mask[roi > 0] = 0

    all_roi += roi

    return new_mask, (all_roi > 0)


def copy_cat_mouth(source_mask, target_mask):
    h, w = source_mask.shape[:2]

    new_mask = copy.deepcopy(source_mask)
    ids = cat_semantic_ids["mouth"]

    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 7

    ref_mask = copy.deepcopy(target_mask)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)
    ref_mask = ref_mask * roi
    bbox_org = mask_to_bbox(delete)
    bbox_ref = mask_to_bbox(roi)
    ratio = ((bbox_ref[2] - bbox_ref[0])) / float((bbox_org[2] - bbox_org[0]))
    A = cv2.getRotationMatrix2D(((bbox_ref[0] + bbox_ref[2]) / 2, (bbox_ref[3] + bbox_ref[1]) / 2), 0, 1.1 / ratio)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    center_org = [(bbox_org[2] + bbox_org[0]) / 2., (bbox_org[1] + bbox_org[3]) / 2.]
    center_ref = [(bbox_ref[2] + bbox_ref[0]) / 2., (bbox_ref[1] + bbox_ref[3]) / 2.]

    A = np.float32([[1, 0, center_org[0] - center_ref[0]], [0, 1, center_org[1] - center_ref[1]]])

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    A = cv2.getRotationMatrix2D(((bbox_org[0] + bbox_org[2]) / 2, (bbox_org[3] + bbox_org[1]) / 2), 0, 1.5)
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi


def enlarge_cat_mouth(source_mask, scale):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    ref_mask = ref_mask
    roi = (ref_mask == 10)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 10)] = 10
    all_roi += roi

    roi = (new_mask == 10)

    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_cat_nose(source_mask, scale):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    ref_mask = ref_mask
    roi = (ref_mask == 12)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 12)] = 12
    all_roi += roi

    roi = (new_mask == 10)

    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_cat_eyes(source_mask, scale):
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, int(new_mask.shape[1] / 2):] = 1

    ref_mask = ref_mask * half_mask
    roi = (ref_mask == 9)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 9)] = 9

    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[:, :int(new_mask.shape[1] / 2)] = 1

    ref_mask = ref_mask * half_mask
    roi = (ref_mask == 9)
    roi = (roi > 0)

    new_mask[roi > 0] = 7

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    new_mask[(ref_mask == 9)] = 9

    roi = (new_mask == 9)

    all_roi += roi

    all_roi = (all_roi > 0)
    roi = all_roi

    return new_mask, roi


################################ Face ################################

semantic_ids = {"changeNose": [26, 27, 28, 29, 30],
                "wideNose": [26, 27, 28, 29, 30],
                "gaze": [9, 10, 39, 40],
                "eyebrow": [14],
                "eyebrowboth": [14, 44],
                "smileWrinkle": [33],
                "wrinkle": [33],
                "pred": [21, 22, 23, 24],
                "mustache": [20],
                "eyes": [7, 8, 9, 10, 11, 12, 13,
                         37, 38, 39, 40, 41, 42, 43],
                "oneEye": [7, 8, 9, 10, 11, 12, 13],
                "smile": [21, 22, 23, 24],
                "openMouth": [21, 22, 23, 24],
                "iris": [10, 40],
                "hair": [17]

                }


def shrink_eyebrow(source_mask, scale):
    ids = semantic_ids['eyebrowboth']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = copy.deepcopy(ref_mask * 0.)

    roi += (ref_mask == ids[0])
    ref_mask = ref_mask * roi

    roi = (roi > 0)
    new_mask[roi > 0] = 1

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[(ref_mask == id)] = id

    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)

    roi = copy.deepcopy(ref_mask * 0.)

    roi += (ref_mask == ids[1])

    ref_mask = ref_mask * roi

    new_mask[roi > 0] = 1

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[(ref_mask == id)] = id

    all_roi += roi

    roi = copy.deepcopy(ref_mask * 0.)
    for id in ids:
        roi += (ref_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def enlarge_iris(source_mask, scale):
    ids = semantic_ids['iris']
    new_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    h, w = new_mask.shape[:2]

    ref_mask = copy.deepcopy(source_mask)

    roi = copy.deepcopy(ref_mask * 0.)

    roi += (ref_mask == ids[0])
    ref_mask = ref_mask * roi

    roi = (roi > 0)
    new_mask[roi > 0] = ids[0] - 1

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[(ref_mask == id)] = id

    all_roi += roi

    ref_mask = copy.deepcopy(source_mask)

    roi = copy.deepcopy(ref_mask * 0.)

    roi += (ref_mask == ids[1])

    ref_mask = ref_mask * roi

    new_mask[roi > 0] = ids[1] - 1

    bbox = mask_to_bbox(roi)

    A = cv2.getRotationMatrix2D(((bbox[0] + bbox[2]) / 2, (bbox[3] + bbox[1]) / 2), 0, scale)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[(ref_mask == id)] = id

    all_roi += roi

    roi = copy.deepcopy(ref_mask * 0.)
    for id in ids:
        roi += (ref_mask == id)
    all_roi += roi

    return new_mask, (all_roi > 0)


def copy_mouth(source_mask, target_mask):
    h, w = source_mask.shape[:2]

    new_mask = copy.deepcopy(source_mask)
    ids = semantic_ids["smile"]

    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 1

    ref_mask = copy.deepcopy(target_mask)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)
    ref_mask = ref_mask * roi
    bbox_org = mask_to_bbox(delete)
    bbox_ref = mask_to_bbox(roi)
    ratio = ((bbox_ref[2] - bbox_ref[0])) / float((bbox_org[2] - bbox_org[0]))
    A = cv2.getRotationMatrix2D(((bbox_ref[0] + bbox_ref[2]) / 2, (bbox_ref[3] + bbox_ref[1]) / 2), 0, 1.1 / ratio)

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    center_org = [(bbox_org[2] + bbox_org[0]) / 2., (bbox_org[1] + bbox_org[3]) / 2.]
    center_ref = [(bbox_ref[2] + bbox_ref[0]) / 2., (bbox_ref[1] + bbox_ref[3]) / 2.]

    A = np.float32([[1, 0, center_org[0] - center_ref[0]], [0, 1, center_org[1] - center_ref[1]]])

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi


def paste_nose(source_mask, target_mask):
    new_mask = copy.deepcopy(source_mask)
    ids = semantic_ids["change_nose"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 1

    ref_mask = copy.deepcopy(target_mask)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi


def wide_nose(source_mask, factor):
    h, w = source_mask.shape[:2]

    new_mask = copy.deepcopy(source_mask)
    ids = semantic_ids["change_nose"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 1

    target_mask = copy.deepcopy(source_mask)
    target_mask[delete == 0] = 0

    target_mask_res = cv2.resize(target_mask, (int(factor * w), h), interpolation=cv2.INTER_NEAREST)

    target_mask_res = target_mask_res[:,
                      int(target_mask_res.shape[1] / 2 - w / 2): int(target_mask_res.shape[1] / 2 + w / 2)]

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (target_mask_res == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(target_mask_res == id)] = id

    return new_mask, roi


def gaze_position(source_mask):
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)
    target_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    roi = (target_mask == 9) + (target_mask == 10)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 11
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, (bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)
    new_mask[target_mask == 9] = 9
    new_mask[target_mask == 10] = 10

    target_mask = copy.deepcopy(source_mask)
    roi = (target_mask == 39) + (target_mask == 40)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 41
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, (bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)
    new_mask[target_mask == 39] = 39
    new_mask[target_mask == 40] = 40

    roi = (new_mask == 9) + (new_mask == 10) + (new_mask == 39) + (new_mask == 40)
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def gaze_position_2(source_mask):
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)

    mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))

    ids = [9, 10, 11, 39, 40, 41]

    for id in ids:
        mask[(new_mask == id)] = 1

    target_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    roi = (target_mask == 9) + (target_mask == 10)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 11
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, 0], [0, 1, (bbox[3] - bbox[1]) / 2]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0) * mask
    new_mask[target_mask == 9] = 9
    new_mask[target_mask == 10] = 10

    target_mask = copy.deepcopy(source_mask)

    roi = (target_mask == 39) + (target_mask == 40)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 41
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, 0], [0, 1, (bbox[3] - bbox[1]) / 2]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0) * mask
    new_mask[target_mask == 39] = 39
    new_mask[target_mask == 40] = 40

    roi = (new_mask == 9) + (new_mask == 10) + (new_mask == 39) + (new_mask == 40)
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def gaze_position_3(source_mask):
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)
    target_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    roi = (target_mask == 9) + (target_mask == 10)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 11
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, - (bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)
    new_mask[target_mask == 9] = 9
    new_mask[target_mask == 10] = 10

    target_mask = copy.deepcopy(source_mask)
    roi = (target_mask == 39) + (target_mask == 40)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 41
    target_mask[roi == 0] = 0
    bbox = mask_to_bbox(roi)
    A = np.float32([[1, 0, (bbox[2] - bbox[0]) / 2], [0, 1, 0]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)
    new_mask[target_mask == 39] = 39
    new_mask[target_mask == 40] = 40

    roi = (new_mask == 9) + (new_mask == 10) + (new_mask == 39) + (new_mask == 40)
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def rise_both_eyebrow(source_mask):
    ids = semantic_ids['eyebrowboth']
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)
    target_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    for id in ids:
        roi += (target_mask == id)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 1
    target_mask[roi == 0] = 0
    A = np.float32([[1, 0, 0], [0, 1, -20]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        new_mask[target_mask == id] = id

    for id in ids:
        roi += (new_mask == id)
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def rise_eyebrow(source_mask):
    h, w = source_mask.shape[:2]
    new_mask = copy.deepcopy(source_mask)
    target_mask = copy.deepcopy(source_mask)
    all_roi = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    roi = (target_mask == 14)
    roi = (roi > 0)
    all_roi += roi

    new_mask[roi > 0] = 1
    target_mask[roi == 0] = 0
    A = np.float32([[1, 0, 0], [0, 1, -20]])
    target_mask = cv2.warpAffine(target_mask.astype(np.uint8), A, (w, h), borderValue=0)
    new_mask[target_mask == 14] = 14

    roi = (new_mask == 14)
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def add_hair(source_mask, target_mask):
    new_mask = copy.deepcopy(source_mask)
    ids = semantic_ids["hair"]
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)

    delete = delete.astype(np.uint8)

    new_mask[delete > 0] = 1

    ref_mask = copy.deepcopy(target_mask)

    roi = copy.deepcopy(target_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi


def delete_mustache(source_mask):
    new_mask = copy.deepcopy(source_mask)

    all_roi = copy.deepcopy(source_mask * 0.)

    ref_mask = copy.deepcopy(source_mask)
    roi = (ref_mask == 20)
    roi = (roi > 0)
    new_mask[roi] = 1
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def delete_wrinkle(source_mask):
    new_mask = copy.deepcopy(source_mask)

    half_mask = np.zeros((source_mask.shape[0], source_mask.shape[1]))
    half_mask[:200, :] = 1
    all_roi = copy.deepcopy(source_mask * 0.)

    ref_mask = half_mask * copy.deepcopy(source_mask)
    roi = (ref_mask == 33)
    roi = (roi > 0)
    new_mask[roi] = 15
    all_roi += roi

    half_mask = np.zeros((source_mask.shape[0], source_mask.shape[1]))
    half_mask[200:, :] = 1

    ref_mask = half_mask * copy.deepcopy(source_mask)
    roi = (ref_mask == 33)
    roi = (roi > 0)
    new_mask[roi] = 1
    all_roi += roi

    all_roi = (all_roi > 0)

    return new_mask, all_roi


def add_smile_wrinkle(source_mask):
    ref_mask = np.load("/data/datasetGAN_face/datasetGAN/training_data/face_processed/image_mask0.npy")
    ref_mask = new_mask = cv2.resize(np.squeeze(ref_mask), dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    half_mask = np.zeros((new_mask.shape[0], new_mask.shape[1]))
    half_mask[300:, :] = 1

    new_mask = copy.deepcopy(source_mask)

    hair = 1 - (new_mask == 17)

    ref_mask = half_mask * ref_mask * hair
    roi = (ref_mask == 33)
    roi = (roi > 0)
    new_mask[roi] = 33

    return new_mask, roi


def close_eyes(source_mask):
    greenscreen_exp_path = "/home/linghuan/ngccli/3D-SDN-mount/styleganSeg/vis_results/greenscreen_encoder/"

    with open(greenscreen_exp_path + 'seg_test.npy', 'rb') as f:
        greenscreen_pred_mask = np.load(f)

    ref_mask = greenscreen_pred_mask[5]

    eyes_mask = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    ids = [7, 8, 9, 10, 11, 12, 13, 37, 38, 39, 40, 41, 42, 43]

    for id in ids:
        eyes_mask[(ref_mask == id)] = id

    new_mask = copy.deepcopy(source_mask)
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)
    delete = delete.astype(np.uint8)
    new_mask[delete > 0] = 1
    ref_mask = copy.deepcopy(eyes_mask)
    roi = copy.deepcopy(eyes_mask * 0.)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi


def close_ono_eyes(source_mask):
    h, w = source_mask.shape[:2]

    greenscreen_exp_path = "/home/linghuan/ngccli/3D-SDN-mount/styleganSeg/vis_results/greenscreen_encoder/"

    with open(greenscreen_exp_path + 'seg_test.npy', 'rb') as f:
        greenscreen_pred_mask = np.load(f)

    ref_mask = greenscreen_pred_mask[5]

    eyes_mask = np.zeros((ref_mask.shape[0], ref_mask.shape[1]))

    ids = [7, 8, 9, 10, 11, 12, 13]

    for id in ids:
        eyes_mask[(ref_mask == id)] = id

    new_mask = copy.deepcopy(source_mask)
    delete = copy.deepcopy(source_mask * 0.)
    for id in ids:
        delete += (new_mask == id)

    delete = (delete > 0)
    delete = delete.astype(np.uint8)
    new_mask[delete > 0] = 1
    ref_mask = copy.deepcopy(eyes_mask)
    roi = copy.deepcopy(eyes_mask * 0.)

    A = np.float32([[1, 0, 2], [0, 1, 6]])

    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), A, (w, h), borderValue=0)

    for id in ids:
        roi += (ref_mask == id)

    roi = (delete + roi) > 0

    for id in ids:
        new_mask[(ref_mask == id)] = id

    return new_mask, roi
