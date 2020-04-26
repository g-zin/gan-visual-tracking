import argparse
import json
import sys
from collections import OrderedDict

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from PIL import Image

from scripts.butil.seq_config import load_seq_config
from scripts.model.result import Result
from config import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sequence", default="Basketball")
    parser.add_argument("-e", "--evaltype", default="OPE")
    parser.add_argument("-t", "--tracker", default="ALL")
    parser.add_argument("--save", action='store_true', default=False)
    args = parser.parse_args()

    #assert args.evaltype in ["OPE", "SRE", "TRE"], "Invalid evaltype: {}".format(args.evaltype)

    base_src = RESULT_SRC.format(args.evaltype)
    trackers = os.listdir(base_src)

    assert args.tracker == "ALL" or args.tracker in trackers, "Invalid tracker: {}".format(args.tracker)

    selected_trackers = [args.tracker]
    if args.tracker == "ALL":
        selected_trackers = trackers

    results = OrderedDict()
    start_frame = None
    seq = None
    for t in selected_trackers:
        src = os.path.join(base_src, t)
        seq_name = [x for x in os.listdir(src) if x == "{}.json".format(args.sequence)][0]
        with open(os.path.join(src, seq_name)) as fh:
            j = json.load(fh)
            result = Result(**j[0])
        if not start_frame:
            start_frame = result.startFrame
        elif start_frame != result.startFrame:
            print("Skipping {} due to incompatible start_frame".format(t))
            continue
        if not seq:
            seq = load_seq_config(result.seqName)
        elif seq.name != result.seqName:
            print("Skipping {} due to incompatible sequence".format(t))
            continue

        results[t] = result.res
    view_result(seq, results, start_frame, args.save)


def view_result(seq, results, startIndex, save):
    fig = plt.figure()
    sub = fig.add_subplot(111)

    src = os.path.join(SEQ_SRC, seq.name)
    image = Image.open(src + '/img/{0:04d}.jpg'.format(startIndex)).convert('RGB')
    im = plt.imshow(image, zorder=0)
    txt = sub.text(0, 0.1, '#{}'.format(seq.startFrame), transform=sub.transAxes, ha="left", color='white')
    txt2 = sub.text(0.5, 1, '{}'.format(", ".join(seq.attributes)), transform=sub.transAxes, ha="center", color='black', fontsize=12)

    LINE_COLORS = ['b', 'r', 'c', 'm', 'y', 'k', '#880015', '#FF7F27', '#00A2E8']
    tracker_color = {}
    legends = [sub.text(1, 0, 'Ground truth', transform=sub.transAxes, ha="right", color="#00ff00")]
    for i, t in enumerate(results.keys()):
        tracker_color[t] = LINE_COLORS[i]
        legends.append(sub.text(1, (i+1)/15, '{}'.format(t), transform=sub.transAxes, ha="right", color=LINE_COLORS[i]))
    txt.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='black')])
    txt2.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white')])
    for l in legends:
        l.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='w')])
    gtRect = None
    t_rect = {}
    n_frames = 0

    def update(num):
        i = startIndex + num
        image = Image.open(src + '/img/{0:04d}.jpg'.format(i)).convert('RGB')
        im.set_data(image)
        txt.set_text('#{}'.format(num))
        g = seq.gtRect[num + startIndex - seq.startFrame]
        gx, gy, gw, gh = get_coordinate(g)
        gtRect.set_xy((gx, gy))
        gtRect.set_width(gw)
        gtRect.set_height(gh)

        for t, res in results.items():
            r = res[num]
            x, y, w, h = get_coordinate(r)
            t_rect[t].set_xy((x, y))
            t_rect[t].set_width(w)
            t_rect[t].set_height(h)
        return tuple([im, txt, txt2, gtRect] + list(t_rect.values()) + legends)

    for t, res in results.items():
        x, y, w, h = get_coordinate(res[0])
        t_rect[t] = plt.Rectangle((x, y), w, h,
                                  linewidth=2, edgecolor=tracker_color[t], zorder=1, fill=False)
        plt.gca().add_patch(t_rect[t])
        if not gtRect:
            gx, gy, gw, gh = get_coordinate(seq.gtRect[startIndex - seq.startFrame])
            gtRect = plt.Rectangle((gx, gy), gw, gh,
                                   linewidth=2, edgecolor="#00ff00", zorder=1, fill=False)
            plt.gca().add_patch(gtRect)
            n_frames = len(res)

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=10, blit=True)
    plt.axis("off")
    if save:
        print("Saving to disk..")
        ani.save('{}.mp4'.format(seq.name), fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()


def get_coordinate(res):
    return int(res[0]), int(res[1]), int(res[2]), int(res[3])


if __name__ == '__main__':
    main()
