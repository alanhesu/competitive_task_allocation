import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import itertools

configs = [
    '(5, 2)',
    '(10, 2)',
    '(20, 2)',
    '(5, 5)',
    '(10, 5)',
    '(20, 5)'
]

def plot_bar(fname, xvar_list, yvar):
    df = pd.read_csv(fname)

    mean_vals = {x: [] for x in configs}

    xvar_combo = []
    for xvar in xvar_list:
        xvals = df[xvar]
        xvar_labels = list(set(list(xvals)))
        xvar_combo.append(xvar_labels)

    labels = list(itertools.product(*xvar_combo))
    for key in mean_vals:
        colname = '{} {}'.format(yvar, key)
        for label in labels:
            dat = df.loc[:, colname]
            for i, xval in enumerate(label):
                dat = dat.loc[df[xvar_list[i]] == xval]
                # dat = df.loc[df[xvar] == label, colname]
            mean_vals[key].append(np.mean(dat))

    x = np.arange(len(labels))
    width = .9/len(configs)

    fig, ax = plt.subplots()
    for i, config in enumerate(configs):
        rects = ax.bar(x - .25 + i*width + width/2, mean_vals[config], width, label='{} {}'.format(yvar, config))
        ax.bar_label(rects, padding=3)

    ax.set_ylabel(yvar)
    ax.set_title('averaged {} for {}'.format(yvar, xvar_list))
    ax.set_xticks(x, labels)
    ax.legend(loc='best')

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--param', type=str, nargs='+', required=True)

    args = parser.parse_args()

    plot_bar(args.fname, args.param, args.metric)
