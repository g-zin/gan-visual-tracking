import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from ..gnet.g_init import NetG
from ..gnet.g_pretrain import g_pretrain
from ..modules.bbreg import BBRegressor
from ..modules.data_prov import RegionExtractor
from ..modules.model import MDNet, BCELoss
from ..modules.sample_generator import SampleGenerator
from ..modules.utils import set_optimizer, overlap_ratio, set_optimizer_g


def forward_samples(model, image, samples, out_layer='conv3', **opts):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def run_vital(**opts):
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

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # init model
    model = MDNet(opts['model_path'])
    model_g = NetG()
    if opts['use_gpu']:
        model = model.cuda()
        model_g = model_g.cuda()

    # Init criterion and optimizer
    criterion = BCELoss()
    #criterion_g = nn.MSELoss(reduction='sum')
    criterion_g = nn.MSELoss(reduction='mean')
    model.set_learnable_params(opts['ft_layers'])
    model_g.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
        target_bb, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
        SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
            target_bb, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
        SampleGenerator('whole', image.size)(
            target_bb, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples, **opts)
    neg_feats = forward_samples(model, image, neg_examples, **opts)

    # Initial training
    train(model, None, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'], **opts)
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()
    g_pretrain(model, model_g, criterion_g, pos_feats, **opts)
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'],
                                     opts['aspect_bbreg'])(
        target_bb, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples, **opts)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bb)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bb, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples, **opts)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = opts['savefig_dir'] != ''
    if opts['visualize'] or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

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
    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        samples = sample_generator(target_bb, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6', **opts)

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bb = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bb = target_bb.mean(axis=0)
        success = target_score > 0

        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(model, image, bbreg_samples, **opts)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bb

        # Save result
        result[i] = target_bb
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bb, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples, **opts)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bb, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples, **opts)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, None, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], **opts)

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, model_g, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'], **opts)

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if opts['visualize'] or savefig:
            im.set_data(image)

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
                fig.savefig(os.path.join(opts['savefig_dir'], '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                  .format(i + 1, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                  .format(i + 1, len(img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list) / spf_total
    return result, result_bb, fps


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

    for i in range(maxiter):

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
        batch_pos_feats = pos_feats[pos_cur_idx]
        if model_g is not None:
            batch_asdn_feats = pos_feats.index_select(0, pos_cur_idx)
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

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
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
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
