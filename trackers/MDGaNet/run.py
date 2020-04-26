import argparse
import json
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
    result = dict()

    img_list, gt = get_seq_data(seq_home, seq_name, opts['set_type'])
    result_nobb, result_bb, fps = tracker.run_vital(img_list=img_list, init_bb=gt[0], gt=gt, **opts)

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
