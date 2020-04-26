import os

import numpy as np
from torch import optim, nn


def overlap_ratio(rect1, rect2):
    """
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    """

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def samples2maskroi(samples, receptive_field, cshape, padded_scene_size, padding_ratio):
    # rois is from domain of original image axis
    # receptive field can be subtracted to x2,y2

    # ratios between original image and resized_image
    cur_resize_ratio = cshape / padded_scene_size
    rois = np.copy(samples)

    # xywh -> x1y1x2y2
    rois[:, 2:4] += rois[:, 0:2]

    # padding application
    rois_paddings = (rois[:, 2:4] - rois[:, 0:2]) * (padding_ratio - 1.) / 2.
    rois[:, 0:2] -= rois_paddings
    rois[:, 2:4] += rois_paddings

    rois[:, 0] *= cur_resize_ratio[0]
    rois[:, 1] *= cur_resize_ratio[1]
    rois[:, 2] = np.maximum(rois[:, 0] + 1, rois[:, 2] * cur_resize_ratio[0] - receptive_field)
    rois[:, 3] = np.maximum(rois[:, 1] + 1, rois[:, 3] * cur_resize_ratio[1] - receptive_field)

    return rois


def set_optimizer(model, lr_base, lr_mult, momentum, w_decay):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def get_seq_data(seq_home, seq_name, set_type):
    # ground truth
    gt = None
    seq_path = os.path.join(seq_home, seq_name)
    img_path = os.path.join(seq_path, 'img')
    if set_type == 'OTB':
        img_list = sorted(
            [os.path.join(img_path, img) for img in os.listdir(img_path) if os.path.splitext(img)[1] == '.jpg'])
        # hacks
        gt_name = ''
        if os.path.exists(os.path.join(seq_path, 'groundtruth_rect.txt')):
            gt_name = 'groundtruth_rect.txt'
        elif os.path.exists(os.path.join(seq_path, 'groundtruth_rect.1.txt')):
            gt_name = 'groundtruth_rect.1.txt'
        elif os.path.exists(os.path.join(seq_path, 'groundtruth_rect.2.txt')):
            gt_name = 'groundtruth_rect.2.txt'

        try:
            gt = np.loadtxt(os.path.join(seq_path, gt_name))
        except:
            gt = np.loadtxt(os.path.join(seq_path, gt_name), delimiter=',')

        if seq_name == 'David':
            img_list = img_list[299:]
        if seq_name == 'Football1':
            img_list = img_list[0:74]
        if seq_name == 'Freeman3':
            img_list = img_list[0:460]
        if seq_name == 'Freeman4':
            img_list = img_list[0:283]
        if seq_name == 'Diving':
            img_list = img_list[0:215]
        if seq_name == 'Tiger1':
            img_list = img_list[5:]
    elif set_type == 'VOT/2016':
        img_list = sorted([os.path.join(seq_path, p) for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(os.path.join(seq_path, 'groundtruth.txt'), delimiter=',')

    # polygon to rect
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list, gt


def crop_image(img, bbox, img_size=[107, 107], padding=16, valid=False):
    # img_size = [w,h]
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size[0]
        pad_h = padding * h / img_size[1]
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if valid:
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(img_w, max_x)
        max_y = min(img_h, max_y)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled = np.array(cropped).resize((img_size[1], img_size[0]))
    return scaled


def append_params(params, module, prefix):
    for child in module.children():
        for k, p in child.named_parameters():
            if p is None:
                continue
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % name)


def set_optimizer_g(model_g, lr, momentum, w_decay):
    params = model_g.get_learnable_params()
    param_list = []
    for k, p in params.items():
        param_list.append(p)

    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer
