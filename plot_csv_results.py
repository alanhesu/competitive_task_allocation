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

def plot_bar(fname, xvar_list, yvar, compare_relative, compare_optimal):
    df = pd.read_csv(fname)

    mean_vals = {x: [] for x in configs}

    xvar_combo = []
    for xvar in xvar_list:
        xvals = df[xvar]
        xvar_labels = list(set(list(xvals)))
        xvar_combo.append(xvar_labels)

    labels = list(itertools.product(*xvar_combo))
    labels = sorted(labels)
    for key in mean_vals:
        colname = '{} {}'.format(yvar, key)
        for label in labels:
            dat = df.loc[:, colname]
            for i, xval in enumerate(label):
                dat = dat.loc[df[xvar_list[i]] == xval]
                # dat = df.loc[df[xvar] == label, colname]
            mean_vals[key].append(np.mean(dat))

    if (compare_optimal):
        optimal_soln = pd.read_csv('lp_soln_3.csv')
        for key in mean_vals:
            colname = '{} {}'.format(yvar, key)
            optimal_val = optimal_soln.loc[:,colname]
            mean_vals[key] = np.array(mean_vals[key])/float(optimal_val)

    mean_val_ratio = {}
    for key in mean_vals:
        mean_val_ratio[key] = np.array(mean_vals[key])/np.nanmax(mean_vals[key])
    if (compare_relative):
        mean_val_plot = mean_val_ratio
    else:
        mean_val_plot = mean_vals

    x = np.arange(len(labels))
    width = .9/len(configs)

    fig, ax = plt.subplots()
    for i, config in enumerate(configs):
        rects = ax.bar(x - .25 + i*width + width/2, mean_val_plot[config], width, label='{} {}'.format(yvar, config))
        ax.bar_label(rects, ['{0:.2f}'.format(x) for x in mean_vals[config]], padding=3)

    ax.set_ylabel(yvar)
    # ax.set_ylim(.5, 1.1)
    ax.set_title('averaged {} for {}'.format(yvar, xvar_list))
    ax.set_xticks(x, labels)
    ax.legend(loc='best')

    fig.tight_layout()

    # print out some aggregate calculations
    meanmean_dict = {}
    for i, label in enumerate(labels):
        meanmean_dict[label] = np.mean([x[i] for x in mean_val_ratio.values()])

    print('average ratios for each category in param(s)')
    print(meanmean_dict)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--metric', type=str, required=True)
    parser.add_argument('--param', type=str, nargs='+', required=True)
    parser.add_argument('--optimal', action='store_true')
    parser.add_argument('--relative', action='store_true')

    args = parser.parse_args()

    plot_bar(args.fname, args.param, args.metric, args.relative, args.optimal)
