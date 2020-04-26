import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from ..gnet.g_pretrain import g_pretrain
from ..gnet.g_init import NetG
from ..modules.bbreg import BBRegressor
from ..modules.image_cropper import ImgCropper
from ..modules.model import MDNet, BinaryLoss
from ..modules.roi_align.modules.roi_align import RoIAlignAdaMax
from ..modules.sample_generator import SampleGenerator, gen_samples
from ..modules.utils import samples2maskroi, overlap_ratio, set_optimizer, set_optimizer_g


def run_mdnet(**opts):
    img_list = opts['img_list']
    gt = opts['gt']
    # init bounding box
    target_bb = np.array(opts['init_bb'])
    # a bounding box per image
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    # first image
    result[0] = np.copy(target_bb)
    result_bb[0] = np.copy(target_bb)

    iou_result = np.zeros((len(img_list), 1))

    # init model
    model = MDNet(opts['model_path'])
    model_g = NetG()
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()
        model_g = model_g.cuda()

    model.set_learnable_params(opts['ft_layers'])
    model_g.set_learnable_params(opts['ft_layers'])

    # init image crop model
    img_crop_model = ImgCropper(1.)
    if opts['use_gpu']:
        img_crop_model.gpu_enable()

    # init criterion and optimizer
    criterion = BinaryLoss()
    #criterion_g = nn.MSELoss(reduction='sum')
    criterion_g = nn.MSELoss(reduction='mean')
    init_optimizer = set_optimizer(model, opts['lr_init'], lr_mult=opts['lr_mult'], momentum=opts['momentum'],
                                   w_decay=opts['w_decay'])
    update_optimizer = set_optimizer(model, opts['lr_update'], lr_mult=opts['lr_mult'], momentum=opts['momentum'],
                                     w_decay=opts['w_decay'])

    tic = time.time()
    # Load first image
    cur_image = Image.open(img_list[0]).convert('RGB')
    cur_image = np.asarray(cur_image)

    # Draw pos/neg samples
    img_shape = cur_image.shape
    pos_examples = gen_samples(SampleGenerator('gaussian', (img_shape[1], img_shape[0]), 0.1, 1.2),
                               target_bb, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_examples = gen_samples(SampleGenerator('uniform', (img_shape[1], img_shape[0]), 1, 2, 1.1),
                               target_bb, opts['n_neg_init'], opts['overlap_neg_init'])
    neg_examples = np.random.permutation(neg_examples)

    cur_bbreg_examples = gen_samples(SampleGenerator('uniform', (img_shape[1], img_shape[0]), 0.3, 1.5, 1.1),
                                     target_bb, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])

    # compute padded sample
    padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
    padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
    padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
    padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
    padded_scene_box = np.reshape(np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)),
                                  (1, 4))

    scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
    if opts['jitter']:
        # horizontal shift
        jittered_scene_box_horizon = np.copy(padded_scene_box)
        jittered_scene_box_horizon[0, 0] -= 4.
        jitter_scale_horizon = 1.

        # vertical shift
        jittered_scene_box_vertical = np.copy(padded_scene_box)
        jittered_scene_box_vertical[0, 1] -= 4.
        jitter_scale_vertical = 1.

        jittered_scene_box_reduce1 = np.copy(padded_scene_box)
        jitter_scale_reduce1 = 1.1 ** (-1)

        # vertical shift
        jittered_scene_box_enlarge1 = np.copy(padded_scene_box)
        jitter_scale_enlarge1 = 1.1 ** (1)

        # scale reduction
        jittered_scene_box_reduce2 = np.copy(padded_scene_box)
        jitter_scale_reduce2 = 1.1 ** (-2)

        # scale enlarge
        jittered_scene_box_enlarge2 = np.copy(padded_scene_box)
        jitter_scale_enlarge2 = 1.1 ** (2)

        scene_boxes = np.concatenate(
            [scene_boxes, jittered_scene_box_horizon, jittered_scene_box_vertical, jittered_scene_box_reduce1,
             jittered_scene_box_enlarge1, jittered_scene_box_reduce2, jittered_scene_box_enlarge2], axis=0)
        jitter_scale = [1., jitter_scale_horizon, jitter_scale_vertical, jitter_scale_reduce1, jitter_scale_enlarge1,
                        jitter_scale_reduce2, jitter_scale_enlarge2]
    else:
        jitter_scale = [1.]

    model.eval()
    for bidx in range(0, scene_boxes.shape[0]):
        crop_img_size = (scene_boxes[bidx, 2:4] * ((opts['img_size'], opts['img_size']) / target_bb[2:4])).astype(
            'int64') * jitter_scale[bidx]
        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(scene_boxes[bidx], (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.

        feat_map = model(cropped_image, out_layer='conv3')

        rel_target_bbox = np.copy(target_bb)
        rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

        batch_num = np.zeros((pos_examples.shape[0], 1))
        cur_pos_rois = np.copy(pos_examples)
        cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0], axis=0)
        scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
        cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                       target_bb[2:4], opts['padding'])
        cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
        cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
        cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
        cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

        batch_num = np.zeros((neg_examples.shape[0], 1))
        cur_neg_rois = np.copy(neg_examples)
        cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0], axis=0)
        cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                       target_bb[2:4], opts['padding'])
        cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
        cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
        cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
        cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

        # bbreg rois
        batch_num = np.zeros((cur_bbreg_examples.shape[0], 1))
        cur_bbreg_rois = np.copy(cur_bbreg_examples)
        cur_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_bbreg_rois.shape[0], axis=0)
        scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
        cur_bbreg_rois = samples2maskroi(cur_bbreg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                         target_bb[2:4], opts['padding'])
        cur_bbreg_rois = np.concatenate((batch_num, cur_bbreg_rois), axis=1)
        cur_bbreg_rois = Variable(torch.from_numpy(cur_bbreg_rois.astype('float32'))).cuda()
        cur_bbreg_feats = model.roi_align_model(feat_map, cur_bbreg_rois)
        cur_bbreg_feats = cur_bbreg_feats.view(cur_bbreg_feats.size(0), -1).data.clone()

        feat_dim = cur_pos_feats.size(-1)

        if bidx == 0:
            pos_feats = cur_pos_feats
            neg_feats = cur_neg_feats
            # bbreg feature
            bbreg_feats = cur_bbreg_feats
            bbreg_examples = cur_bbreg_examples
        else:
            pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
            neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)
            # bbreg feature
            bbreg_feats = torch.cat((bbreg_feats, cur_bbreg_feats), dim=0)
            bbreg_examples = np.concatenate((bbreg_examples, cur_bbreg_examples), axis=0)

    if pos_feats.size(0) > opts['n_pos_init']:
        pos_idx = np.asarray(list(range(pos_feats.size(0))))
        np.random.shuffle(pos_idx)
        pos_feats = pos_feats[pos_idx[0:opts['n_pos_init']], :]
    if neg_feats.size(0) > opts['n_neg_init']:
        neg_idx = np.asarray(list(range(neg_feats.size(0))))
        np.random.shuffle(neg_idx)
        neg_feats = neg_feats[neg_idx[0:opts['n_neg_init']], :]

    # bbreg
    if bbreg_feats.size(0) > opts['n_bbreg']:
        bbreg_idx = np.asarray(list(range(bbreg_feats.size(0))))
        np.random.shuffle(bbreg_idx)
        bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']], :]
        bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']], :]
        # print bbreg_examples.shape

    # open images and crop patch from obj
    extra_obj_size = np.array((opts['img_size'], opts['img_size']))
    extra_crop_img_size = extra_obj_size * (opts['padding'] + 0.6)
    replicateNum = 100
    for iidx in range(replicateNum):
        extra_target_bbox = np.copy(target_bb)

        extra_scene_box = np.copy(extra_target_bbox)
        extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
        extra_scene_box_size = extra_scene_box[2:4] * (opts['padding'] + 0.6)
        extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
        extra_scene_box[2:4] = extra_scene_box_size

        extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
        cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

        extra_scene_box[0] += extra_shift_offset[0]
        extra_scene_box[1] += extra_shift_offset[1]
        extra_scene_box[2:4] *= cur_extra_scale[0]

        scaled_obj_size = float(opts['img_size']) / cur_extra_scale[0]

        cur_extra_cropped_image, _ = img_crop_model.crop_image(cur_image, np.reshape(extra_scene_box, (1, 4)),
                                                               extra_crop_img_size)
        cur_extra_cropped_image = cur_extra_cropped_image.detach()

        cur_extra_pos_examples = gen_samples(SampleGenerator('gaussian', (img_shape[1], img_shape[0]), 0.1, 1.2),
                                             extra_target_bbox, opts['n_pos_init'] // replicateNum,
                                             opts['overlap_pos_init'])
        cur_extra_neg_examples = gen_samples(SampleGenerator('uniform', (img_shape[1], img_shape[0]), 0.3, 2, 1.1),
                                             extra_target_bbox, opts['n_neg_init'] // replicateNum // 4,
                                             opts['overlap_neg_init'])

        # bbreg sample
        cur_extra_bbreg_examples = gen_samples(SampleGenerator('uniform', (img_shape[1], img_shape[0]), 0.3, 1.5, 1.1),
                                               extra_target_bbox, opts['n_bbreg'] // replicateNum // 4,
                                               opts['overlap_bbreg'], opts['scale_bbreg'])

        batch_num = iidx * np.ones((cur_extra_pos_examples.shape[0], 1))
        cur_extra_pos_rois = np.copy(cur_extra_pos_examples)
        cur_extra_pos_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                cur_extra_pos_rois.shape[0], axis=0)
        cur_extra_pos_rois = samples2maskroi(cur_extra_pos_rois, model.receptive_field,
                                             (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                             opts['padding'])
        cur_extra_pos_rois = np.concatenate((batch_num, cur_extra_pos_rois), axis=1)

        batch_num = iidx * np.ones((cur_extra_neg_examples.shape[0], 1))
        cur_extra_neg_rois = np.copy(cur_extra_neg_examples)
        cur_extra_neg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)), cur_extra_neg_rois.shape[0],
                                                axis=0)
        cur_extra_neg_rois = samples2maskroi(cur_extra_neg_rois, model.receptive_field,
                                             (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                             opts['padding'])
        cur_extra_neg_rois = np.concatenate((batch_num, cur_extra_neg_rois), axis=1)

        # bbreg rois
        batch_num = iidx * np.ones((cur_extra_bbreg_examples.shape[0], 1))
        cur_extra_bbreg_rois = np.copy(cur_extra_bbreg_examples)
        cur_extra_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                  cur_extra_bbreg_rois.shape[0], axis=0)
        cur_extra_bbreg_rois = samples2maskroi(cur_extra_bbreg_rois, model.receptive_field,
                                               (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                               opts['padding'])
        cur_extra_bbreg_rois = np.concatenate((batch_num, cur_extra_bbreg_rois), axis=1)

        if iidx == 0:
            extra_cropped_image = cur_extra_cropped_image

            extra_pos_rois = np.copy(cur_extra_pos_rois)
            extra_neg_rois = np.copy(cur_extra_neg_rois)
            # bbreg rois
            extra_bbreg_rois = np.copy(cur_extra_bbreg_rois)
            extra_bbreg_examples = np.copy(cur_extra_bbreg_examples)
        else:
            extra_cropped_image = torch.cat((extra_cropped_image, cur_extra_cropped_image), dim=0)

            extra_pos_rois = np.concatenate((extra_pos_rois, np.copy(cur_extra_pos_rois)), axis=0)
            extra_neg_rois = np.concatenate((extra_neg_rois, np.copy(cur_extra_neg_rois)), axis=0)
            # bbreg rois
            extra_bbreg_rois = np.concatenate((extra_bbreg_rois, np.copy(cur_extra_bbreg_rois)), axis=0)
            extra_bbreg_examples = np.concatenate((extra_bbreg_examples, np.copy(cur_extra_bbreg_examples)), axis=0)

    extra_pos_rois = Variable(torch.from_numpy(extra_pos_rois.astype('float32'))).cuda()
    extra_neg_rois = Variable(torch.from_numpy(extra_neg_rois.astype('float32'))).cuda()
    # bbreg rois
    extra_bbreg_rois = Variable(torch.from_numpy(extra_bbreg_rois.astype('float32'))).cuda()

    extra_cropped_image -= 128.

    extra_feat_maps = model(extra_cropped_image, out_layer='conv3')
    # Draw pos/neg samples
    img_shape = cur_image.shape

    extra_pos_feats = model.roi_align_model(extra_feat_maps, extra_pos_rois)
    extra_pos_feats = extra_pos_feats.view(extra_pos_feats.size(0), -1).data.clone()

    extra_neg_feats = model.roi_align_model(extra_feat_maps, extra_neg_rois)
    extra_neg_feats = extra_neg_feats.view(extra_neg_feats.size(0), -1).data.clone()
    # bbreg feat
    extra_bbreg_feats = model.roi_align_model(extra_feat_maps, extra_bbreg_rois)
    extra_bbreg_feats = extra_bbreg_feats.view(extra_bbreg_feats.size(0), -1).data.clone()

    # concatenate extra features to original_features
    pos_feats = torch.cat((pos_feats, extra_pos_feats), dim=0)
    neg_feats = torch.cat((neg_feats, extra_neg_feats), dim=0)
    # concatenate extra bbreg feats to original_bbreg_feats
    bbreg_feats = torch.cat((bbreg_feats, extra_bbreg_feats), dim=0)
    bbreg_examples = np.concatenate((bbreg_examples, extra_bbreg_examples), axis=0)

    torch.cuda.empty_cache()
    model.zero_grad()

    # Initial training
    train(model, None, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], **opts)
    #del init_optimizer, neg_feats
    if opts['use_gpu']:
        torch.cuda.empty_cache()
    g_pretrain(model, model_g, criterion_g, pos_feats, **opts)
    if opts['use_gpu']:
        torch.cuda.empty_cache()

    # bbreg train
    if bbreg_feats.size(0) > opts['n_bbreg']:
        bbreg_idx = np.asarray(list(range(bbreg_feats.size(0))))
        np.random.shuffle(bbreg_idx)
        bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']], :]
        bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']], :]
    bbreg = BBRegressor((img_shape[1], img_shape[0]))
    bbreg.train(bbreg_feats, bbreg_examples, target_bb)

    if pos_feats.size(0) > opts['n_pos_update']:
        pos_idx = np.asarray(list(range(pos_feats.size(0))))
        np.random.shuffle(pos_idx)
        pos_feats_all = [pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())]
    if neg_feats.size(0) > opts['n_neg_update']:
        neg_idx = np.asarray(list(range(neg_feats.size(0))))
        np.random.shuffle(neg_idx)
        neg_feats_all = [neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())]

    spf_total = time.time() - tic
    # spf_total = 0. # no first frame

    # Visualize
    savefig = opts['savefig_dir'] != ''
    if opts['visualize'] or savefig:
        dpi = 80.0
        figsize = (cur_image.shape[1] / dpi, cur_image.shape[0] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(cur_image, aspect='normal')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if opts['visualize']:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(opts['savefig_dir'], '0000.jpg'), dpi=dpi)

    # Main loop
    trans_f = opts['trans_f']
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        cur_image = Image.open(img_list[i]).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Estimate target bbox
        img_shape = cur_image.shape
        samples = gen_samples(
            SampleGenerator('gaussian', (img_shape[1], img_shape[0]), trans_f, opts['scale_f'], valid=True),
            target_bb, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (padded_scene_box[2:4] * ((opts['img_size'], opts['img_size']) / target_bb[2:4])).astype(
            'int64')
        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.

        model.eval()
        feat_map = model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bb)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, model.receptive_field, (opts['img_size'], opts['img_size']),
                                      target_bb[2:4], opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = model(sample_feats, in_layer='fc4')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.data.cpu().numpy()
        target_score = top_scores.data.mean()
        target_bb = samples[top_idx].mean(axis=0)

        success = target_score > opts['success_thr']

        # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        # bb regression
        if success:
            bbreg_feats = sample_feats[top_idx, :]
            bbreg_samples = samples[top_idx]
            bbreg_samples = bbreg.predict(bbreg_feats.data, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bb

        # Save result
        result[i] = target_bb
        result_bb[i] = bbreg_bbox
        iou_result[i] = 1.

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (img_shape[1], img_shape[0]), 0.1, 1.2), target_bb,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (img_shape[1], img_shape[0]), 1.5, 1.2), target_bb,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (opts['img_size'], opts['img_size']) / target_bb[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = img_crop_model.crop_image(cur_image,
                                                                         np.reshape(scene_boxes[bidx], (1, 4)),
                                                                         crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bb)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bb[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bb[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(list(range(pos_feats.size(0))))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(list(range(neg_feats.size(0))))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, None, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], **opts)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, model_g, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], **opts)

        spf = time.time() - tic
        spf_total += spf

        # Visualize
        if opts['visualize'] or savefig:
            im.set_data(cur_image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if opts['visualize']:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(opts['savefig_dir'], '%04d.jpg' % i), dpi=dpi)

        if opts['visual_log']:
            if gt is None:
                print("Frame %d/%d, Score %.3f, Time %.3f" % (i, len(img_list), target_score, spf))
            else:
                print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                      (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
        iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]

    fps = len(img_list) / spf_total
    # fps = (len(img_list)-1) / spf_total #no first frame
    return iou_result, result_bb, fps, result


def train(model, model_g, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4', **opts):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while len(pos_idx) < batch_pos * maxiter:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while len(neg_idx) < batch_neg_cand * maxiter:
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        if model_g is not None:
            batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()  # model transfer into evaluation mode
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()  # model transfer into train mode

        if model_g is not None:
            model_g.eval()
            res_asdn = model_g(batch_asdn_feats)
            model_g.train()
            num = res_asdn.size(0)
            mask_asdn = torch.ones(num, 512, 3, 3)
            res_asdn = res_asdn.view(num, 3, 3)
            for i in range(num):
                feat_ = res_asdn[i, :, :]
                featlist = feat_.view(1, 9).squeeze()
                feat_list = featlist.detach().cpu().numpy()
                idlist = feat_list.argsort()
                idxlist = idlist[:3]

                for k in range(len(idxlist)):
                    idx = idxlist[k]
                    row = idx // 3
                    col = idx % 3
                    mask_asdn[:, :, col, row] = 0
            mask_asdn = mask_asdn.view(mask_asdn.size(0), -1)
            if opts['use_gpu']:
                batch_asdn_feats = batch_asdn_feats.cuda()
                mask_asdn = mask_asdn.cuda()
            batch_asdn_feats = batch_asdn_feats * mask_asdn

        # forward
        if model_g is None:
            pos_score = model(batch_pos_feats, in_layer=in_layer)
        else:
            pos_score = model(batch_asdn_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if model_g is not None:
            for iter in range(opts['maxiter_g_update']):
                start = time.time()
                prob_k = torch.zeros(9)
                for k in range(9):
                    row = k // 3
                    col = k % 3

                    model.eval()
                    batch = batch_pos_feats.view(batch_pos, 512, 3, 3)
                    batch[:, :, col, row] = 0
                    batch = batch.view(batch.size(0), -1)

                    if opts['use_gpu']:
                        batch = batch.cuda()

                    #score = model(batch, in_layer='fc4')
                    prob = model(batch, in_layer='fc4', out_layer='fc6_softmax')[:, 1]
                    model.train()

                    #prob = F.softmax(score, dim=1)[:, 1]
                    prob_k[k] = prob.sum()

                _, idx = torch.min(prob_k, 0)
                idx = idx.item()
                row = idx // 3
                col = idx % 3

                optimizer_g = set_optimizer_g(model_g, lr=opts['lr_g_update'], momentum=opts['momentum'], w_decay=opts['w_decay'])
                labels = torch.ones(batch_pos, 1, 3, 3)
                labels[:, :, col, row] = 0

                batch_pos_feats = batch_pos_feats.view(batch_pos_feats.size(0), -1)
                res = model_g(batch_pos_feats)
                labels = labels.view(batch_pos, -1)
                #criterion_g = nn.MSELoss(reduction='sum')
                criterion_g = nn.MSELoss(reduction='mean')
                loss_g_2 = criterion_g(res.float(), labels.cuda().float())
                model_g.zero_grad()
                loss_g_2.backward()
                optimizer_g.step()

                #objective = loss_g_2 / batch_pos
                objective = loss_g_2
                end = time.time()
                print('asdn objective %.3f, %.2f s' % (objective, end - start))

        if opts['visual_log']:
            print("Iter %d, Loss %.4f" % (iter, loss.data[0]))
