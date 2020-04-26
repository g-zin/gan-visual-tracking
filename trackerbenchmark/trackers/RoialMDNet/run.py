import argparse
import json
import math
import os

import yaml

STANDALONE = False


def main(custom_opts=None):
    if STANDALONE:
        from src.tracking import tracker
        from src.modules.utils import get_seq_data
    else:
        from .src.tracking import tracker
        from .src.modules.utils import get_seq_data

    with open('src/tracking/options.yaml', 'r') as fh:
        opts = yaml.safe_load(fh)
    opts.update(custom_opts)

    seq_home = opts['data_path']
    seq_name = opts['seq_name']
    iou_list = []
    result = dict()

    img_list, gt = get_seq_data(seq_home, seq_name, opts['set_type'])
    iou_result, result_bb, fps, result_nobb = tracker.run_mdnet(img_list=img_list, init_bb=gt[0], gt=gt, **opts)

    enabled_frame_count = 0.
    for j in range(len(iou_result)):
        if not math.isnan(iou_result[j]):
            enabled_frame_count += 1.
        else:
            # ground_truth is not allowed
            iou_result[j] = 0.

    iou_list.append(iou_result.sum() / enabled_frame_count)

    result['type'] = 'rect'
    result['res'] = result_bb.round().tolist()
    result['fps'] = fps
    print(result)
    #print('saving {}...'.format(os.path.abspath(opts['result_path'])))
    #with open(opts['result_path'], 'w') as fh:
    #    json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    with open('src/tracking/options.yaml', 'r') as fh:
        opts = yaml.safe_load(fh)
    allowed_args = ['-%s' % k for k in opts.keys()]
    parser = argparse.ArgumentParser()
    for a in allowed_args:
        parser.add_argument(a)
    args = parser.parse_args()
    STANDALONE = True
    custom_opts = {k[1:]: getattr(args, k[1:], None) for k in allowed_args}
    main({k: v for k, v in custom_opts.items() if v is not None})
