import math
import pickle
import yaml
import time
import argparse
import numpy as np

from torch import nn
import torch
from sklearn.model_selection import KFold

from .data_prov import RegionDataset
from ..modules.model import MDNet, set_optimizer, BCELoss, Precision


def train_mdnet(opts):

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
        full_dataset = [RegionDataset(seq['images'], seq['gt'], opts) for seq in data.values()]

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
    else:
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
    model = MDNet(opts['init_model_path'], max(map(len, cv_train)))
    #model = MDNet(None, K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    epoch_prec = []
    cv_idx = 0
    # Main training loop
    for i in range(opts['n_cycles']):
        if cv_idx >= opts['cross_val_k']:
            cv_idx = 0
        train_dataset = [full_dataset[i] for i in cv_train[cv_idx]]
        val_dataset = [full_dataset[i] for i in cv_val[cv_idx]]
        cv_idx += 1

        print(('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles'])))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(len(train_dataset))
        k_list = np.random.permutation(len(train_dataset))
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = next(train_dataset[k])
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print(('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc)))

        print(('Mean Precision: {:.3f}'.format(prec.mean())))
        print(('Save model to {:s}'.format(opts['model_path'])))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()

        # validation
        # print("Validation:")
        # model.eval()
        # prec = np.zeros(len(val_dataset))
        # k_list = np.random.permutation(len(val_dataset))
        # for j, k in enumerate(k_list):
        #     tic = time.time()
        #     # training
        #     pos_regions, neg_regions = next(val_dataset[k])
        #     if opts['use_gpu']:
        #         pos_regions = pos_regions.cuda()
        #         neg_regions = neg_regions.cuda()
        #     pos_score = model(pos_regions)
        #     neg_score = model(neg_regions)
        #
        #     prec[k] = evaluator(pos_score, neg_score)
        #
        #     toc = time.time() - tic
        #     print(('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Precision {:.3f}, Time {:.3f}'
        #            .format(i, opts['n_cycles'], j, len(k_list), k, prec[k], toc)))
        #
        # epoch_prec.append(prec.mean())
        # print(('Mean Precision: {:.3f}'.format(epoch_prec[i])))
        # if len(epoch_prec) > 1 and math.fabs(epoch_prec[-1] - epoch_prec[-2]) <= opts['error_threshold']:
        #     print("Achieved desired error threshold: {}<={}".format(math.fabs(epoch_prec[-1] - epoch_prec[-2]), opts['error_threshold']))
        #     break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='imagenet', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    train_mdnet(opts)


if __name__ == "__main__":
    main()
