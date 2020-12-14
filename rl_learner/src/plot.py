import matplotlib.pyplot as plt
import numpy as np


def plot_progress(x_axis, *argv):
    alpha = 0.8
    linewidth = 2
    iters = len(argv) // 2

    plt.rcParams["figure.figsize"] = (8, 5)
    for i in range(iters):
        n = len(argv[i*3])
        stat_mean = np.mean(argv[i*3], axis=0)
        stat_std = np.std(argv[i*3], axis=0)
        ci_95 = 1.96 * stat_std / np.sqrt(n)

        plt.plot(x_axis, stat_mean, argv[i * 3 + 1], label=argv[i * 3 + 2],
                 alpha=alpha, linewidth=linewidth)

        plt.fill_between(x_axis, stat_mean + ci_95, stat_mean -
                         ci_95, alpha=0.5)

    plt.xlabel('Episode number')
    plt.ylabel('Statistic')
    plt.subplots_adjust(bottom=0.30)
    plt.legend(bbox_to_anchor=(0, -0.09, 1, -0.09), loc="upper left",
               mode="expand", borderaxespad=0, ncol=1)
    # plt.legend(loc="lower center")
