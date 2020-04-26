import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import *
from scripts.butil.load_results import load_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph", default="overlap")
    parser.add_argument("-e", "--evaltype", default="OPE")
    parser.add_argument("-t", "--testname", default="tb50")
    parser.add_argument("-a", "--attribute", default="ALL")
    args = parser.parse_args()

    evalTypes = args.evaltype.split(',')
    testname = args.testname
    graph = args.graph
    attrName = args.attribute

    for i in range(len(evalTypes)):
        evalType = evalTypes[i]
        result_src = RESULT_SRC.format(evalType)
        trackers = os.listdir(result_src)
        scoreList = []
        for t in trackers:
            try:
                score = load_scores(evalType, t, testname)
                scoreList.append(score)
            except:
                pass
        if graph == 'precision':
            plt = get_precision_graph(scoreList, i, evalType, testname, attrName)
        elif graph == 'overlap':
            plt = get_overlap_graph(scoreList, i, evalType, testname, attrName)
        else:
            plt = get_fps_graph(scoreList, i, evalType, testname, attrName)
    plt.show()


def get_overlap_graph(scoreList, fignum, evalType, testname, attrName):
    fig = plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    rankList = sorted(scoreList,
                      key=lambda o: sum(o[attrName].successRateList), reverse=True)
    for i in range(len(rankList)):
        result = rankList[i]
        tracker = result[attrName].tracker
        attr = result[attrName]
        if len(attr.successRateList) == len(thresholdSetOverlap):
            if i < MAXIMUM_LINES:
                ls = '-'
                if i % 2 == 1:
                    ls = '--'
                ave = sum(attr.successRateList) / float(len(attr.successRateList))
                plt.plot(thresholdSetOverlap, attr.successRateList,
                         c=LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker, ave), lw=2.0, ls=ls)
            else:
                plt.plot(thresholdSetOverlap, attr.successRateList,
                         label='', alpha=0.5, c='#202020', ls='--')
        else:
            print('err')
    plt.title('{}_{} ({})'.format(evalType, testname.upper(), attrName))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('Overlap Threshold')
    plt.ylabel('Success Rate')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    plt.show()
    return plt


def get_precision_graph(scoreList, fignum, evalType, testname, attrName):
    fig = plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    rankList = sorted(scoreList,
                      key=lambda o: o[attrName].precisionList[20], reverse=True)
    for i in range(len(rankList)):
        result = rankList[i]
        tracker = result[attrName].tracker
        attr = result[attrName]
        if len(attr.precisionList) == len(thresholdSetError):
            if i < MAXIMUM_LINES:
                ls = '-'
                if i % 2 == 1:
                    ls = '--'
                plt.plot(thresholdSetError, attr.precisionList,
                         c=LINE_COLORS[i], label='{0} [{1:.2f}]'.format(tracker, attr.precisionList[20]), lw=2.0, ls=ls)
            else:
                plt.plot(thresholdSetError, attr.precisionList,
                         label='', alpha=0.5, c='#202020', ls='--')
        else:
            print('err')
    plt.title('{}_{} ({})'.format(evalType, testname.upper(), attrName))
    plt.rcParams.update({'axes.titlesize': 'medium'})
    plt.xlabel('Location Error Threshold')
    plt.ylabel('Precision')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    plt.show()
    return plt


def get_fps_graph(scoreList, fignum, evalType, testname, attrName):
    fig = plt.figure(num=fignum, figsize=(9, 6), dpi=70)
    rankList = sorted(scoreList,
                      key=lambda o: o[attrName].fps, reverse=True)
    trackers = [r[attrName].tracker for r in rankList]
    x_pos = np.arange(len(trackers))
    fpsList = [r[attrName].fps for r in rankList]

    plt.bar(x_pos, fpsList, align='center', alpha=0.8)
    plt.xticks(x_pos, trackers)
    plt.ylabel('FPS')
    plt.title('{}_{} ({})'.format(evalType, testname.upper(), attrName))
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.xlim([min(x_pos) - 0.5, max(x_pos) + 0.5])
    plt.grid(color='#101010', alpha=0.5, ls=':')
    for i, v in enumerate(fpsList):
        plt.text(i, v+0.2, "{:1.2f}".format(v), color='blue', va='center', ha='center', fontweight='bold')
    #plt.legend(fontsize='medium')
    # plt.savefig(BENCHMARK_SRC + 'graph/{0}_sq.png'.format(evalType), dpi=74, bbox_inches='tight')
    plt.show()
    return plt


if __name__ == '__main__':
    main()
