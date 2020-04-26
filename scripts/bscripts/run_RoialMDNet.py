import json

from config import *
from trackers.RoialMDNet.run import main

TRACKER_PATH = os.path.join(TRACKER_SRC, 'RoialMDNet')
TRACKER_PATH = os.path.abspath(TRACKER_PATH)


def run_RoialMDNet(seq, rp, bSaveImage):
    seq_name = seq.name[:seq.name.rfind('_')]
    tmp_res = os.path.join(TRACKER_PATH, 'tmp_res_{}.json'.format(seq_name))
    data_path = os.path.abspath(SEQ_SRC)

    curdir = os.path.abspath(os.getcwd())
    os.chdir(TRACKER_PATH)
    print("data_path={}".format(data_path))
    print("seq_name={}".format(seq_name))
    #print("result_path={}".format(tmp_res))
    res = main({'result_path': tmp_res, 'data_path': data_path, 'seq_name': seq_name})
    os.chdir(curdir)

    #res = json.load(open(tmp_res, 'r'))
    #os.remove(tmp_res)
    return res

