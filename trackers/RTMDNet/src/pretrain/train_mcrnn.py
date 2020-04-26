import argparse
import math
import pickle
import time

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold
from torch import nn
from torch.autograd import Variable

from ..modules.data_prov import RegionDataset
from ..modules.model import MDNet, BinaryLoss, Precision
from ..modules.roi_align.modules.roi_align import RoIAlignAdaMax
from ..modules.utils import set_optimizer


def train_mdnet(opts):

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
        full_dataset = [RegionDataset(seq['images'], seq['gt'], opts['receptive_field'], opts) for seq in data.values()]

    # init cross validation
    cv_train = [list(range(len(full_dataset)))]  # all dataset for training
    cv_val = [[]]  # no validation set
    if opts['val_data_path']:  # separate dataset for validation (no cross)
        with open(opts['val_data_path'], 'rb') as fp:
            val_data = pickle.load(fp)
        # indices follow training indices
        cv_val = [list(range(len(full_dataset), len(full_dataset) + len(val_data)))]
        full_dataset.extend([RegionDataset(seq['images'], seq['gt'], opts) for seq in val_data.values()])
        print("Validating with separate dataset.")
    elif opts['cross_val_k'] > 0:
        # k-fold
        print("Cross-validating using {} groups".format(opts['cross_val_k']))
        cv = KFold(n_splits=opts['cross_val_k'], random_state=0)
        cv_train = []
        cv_val = []
        for train_indices, val_indices in cv.split(full_dataset):
            print("  Group 1: {} Tr, {} Va".format(len(train_indices), len(val_indices)))
            cv_train.append(train_indices)
            cv_val.append(val_indices)

    # Init model
    model = MDNet(opts['init_model_path'], max(map(len, cv_train)), opts['receptive_field'])
    #model = MDNet(None, K)
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    binaryCriterion = BinaryLoss()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], lr_mult=opts['lr_mult'],
                              momentum=opts['momentum'],
                              w_decay=opts['w_decay'])

    best_score = 0.
    batch_cur_idx = 0
    epoch_prec = []
    cv_idx = 0
    for i in range(opts['n_cycles']):
        if cv_idx >= opts['cross_val_k']:
            cv_idx = 0
        train_dataset = [full_dataset[i] for i in cv_train[cv_idx]]
        val_dataset = [full_dataset[i] for i in cv_val[cv_idx]]
        cv_idx += 1

        print("==== Start Cycle %d ====" % (i))

        #training
        model.train()
        K = len(train_dataset)
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        totalTripleLoss = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            try:
                cropped_scenes, pos_rois, neg_rois = next(train_dataset[k])
            except Exception as e:
                print("Error reading next scenes: {}".format(e))
                continue

            try:
                for sidx in range(0, len(cropped_scenes)):
                    cur_scene = cropped_scenes[sidx]
                    cur_pos_rois = pos_rois[sidx]
                    cur_neg_rois = neg_rois[sidx]

                    cur_scene = Variable(cur_scene)
                    cur_pos_rois = Variable(cur_pos_rois)
                    cur_neg_rois = Variable(cur_neg_rois)
                    if opts['use_gpu']:
                        cur_scene = cur_scene.cuda()
                        cur_pos_rois = cur_pos_rois.cuda()
                        cur_neg_rois = cur_neg_rois.cuda()
                    cur_feat_map = model(cur_scene, k, out_layer='conv3')

                    cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                    cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                    if sidx == 0:
                        pos_feats = [cur_pos_feats]
                        neg_feats = [cur_neg_feats]
                    else:
                        pos_feats.append(cur_pos_feats)
                        neg_feats.append(cur_neg_feats)
                feat_dim = cur_neg_feats.size(1)
                pos_feats = torch.stack(pos_feats, dim=0).view(-1, feat_dim)
                neg_feats = torch.stack(neg_feats, dim=0).view(-1, feat_dim)
            except Exception as e:
                print("Error forwarding: {}".format(e))
                continue

            pos_score = model(pos_feats, k, in_layer='fc4')
            neg_score = model(neg_feats, k, in_layer='fc4')

            cls_loss = binaryCriterion(pos_score, neg_score)

            # inter frame classification

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:, 1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0), 1))

            K_perm = np.random.permutation(K)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if k == cidx:
                    continue
                else:
                    interclass_score = model(pos_feats, cidx, in_layer='fc4')
                    total_interclass_score = torch.cat((total_interclass_score,
                                                        interclass_score[:, 1].contiguous().view(
                                                            (interclass_score.size(0), 1))), dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            totalInterClassLoss[k] = interclass_loss.data.item()

            (cls_loss + 0.1 * interclass_loss).backward()

            batch_cur_idx += 1
            if (batch_cur_idx % opts['seqbatch_size']) == 0:
                torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0

            # evaulator
            prec[k] = evaluator(pos_score, neg_score)
            # computation latency
            toc = time.time() - tic

            print("Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" %
                  (i, j, k, cls_loss.data.item(), prec[k], totalInterClassLoss[k], toc))

        cur_score = prec.mean()
        #try:
        #    total_miou = sum(total_iou) / len(total_iou)
        #except:
        #    total_miou = 0.
        #print("Mean Precision: %.3f Triple Loss: %.3f Inter Loss: %.3f IoU: %.3f" % (
        #    prec.mean(), cur_triple_loss, totalInterClassLoss.mean(), total_miou))
        print(('Mean Precision: {:.3f}'.format(prec.mean())))
        if cur_score > best_score:
            best_score = cur_score
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers': model.layers.state_dict()}
            print("Save model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()

        # validation
        print("Validation:")
        model.eval()
        K = len(val_dataset)
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        totalTripleLoss = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            try:
                cropped_scenes, pos_rois, neg_rois = next(val_dataset[k])
            except Exception as e:
                print("Error reading next scenes: {}".format(e))
                continue

            try:
                for sidx in range(0, len(cropped_scenes)):
                    cur_scene = cropped_scenes[sidx]
                    cur_pos_rois = pos_rois[sidx]
                    cur_neg_rois = neg_rois[sidx]

                    cur_scene = Variable(cur_scene)
                    cur_pos_rois = Variable(cur_pos_rois)
                    cur_neg_rois = Variable(cur_neg_rois)
                    if opts['use_gpu']:
                        cur_scene = cur_scene.cuda()
                        cur_pos_rois = cur_pos_rois.cuda()
                        cur_neg_rois = cur_neg_rois.cuda()
                    cur_feat_map = model(cur_scene, k, out_layer='conv3')

                    cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                    cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                    if sidx == 0:
                        pos_feats = [cur_pos_feats]
                        neg_feats = [cur_neg_feats]
                    else:
                        pos_feats.append(cur_pos_feats)
                        neg_feats.append(cur_neg_feats)
                feat_dim = cur_neg_feats.size(1)
                pos_feats = torch.stack(pos_feats, dim=0).view(-1, feat_dim)
                neg_feats = torch.stack(neg_feats, dim=0).view(-1, feat_dim)
            except Exception as e:
                print("Error forwarding: {}".format(e))
                continue

            pos_score = model(pos_feats, k, in_layer='fc4')
            neg_score = model(neg_feats, k, in_layer='fc4')

            # evaulator
            prec[k] = evaluator(pos_score, neg_score)
            # computation latency
            toc = time.time() - tic

            print("Cycle %2d, K %2d (%2d), Prec %.3f,Time %.3f" %
                  (i, j, k, prec[k], toc))

        epoch_prec.append(prec.mean())
        print(('Mean Precision: {:.3f}'.format(epoch_prec[i])))
        if opts['cross_val_k'] > 0 and len(epoch_prec) > 1 and math.fabs(epoch_prec[-1] - epoch_prec[-2]) <= opts['error_threshold']:
            print("Achieved desired error threshold: {}<={}".format(math.fabs(epoch_prec[-1] - epoch_prec[-2]), opts['error_threshold']))
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='vot', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    pretrain_opts = yaml.safe_load(open('pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    pretrain_opts['padded_img_size'] = pretrain_opts['img_size'] * int(pretrain_opts['padding_ratio'])
    train_mdnet(pretrain_opts)


if __name__ == "__main__":
    main()
