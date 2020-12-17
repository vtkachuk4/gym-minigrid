import matplotlib.pyplot as plt
import numpy as np


def plot_progress(stat, ep_chunk, line_format, label):

    x_axis = np.linspace(1, len(stat[0]) * ep_chunk, int(len(stat[0])))

    plt.rcParams["figure.figsize"] = (8, 5)
    stat_mean = np.mean(stat, axis=0)
    stat_std = np.std(stat, axis=0)
    ci_95 = 1.96 * stat_std / np.sqrt(len(stat))

    plt.plot(x_axis, stat_mean, line_format, label=label,
             alpha=0.5, linewidth=2)
    plt.fill_between(x_axis, stat_mean + ci_95, stat_mean -
                     ci_95, alpha=0.5)

    plt.xlabel('Episode number')
    plt.subplots_adjust(bottom=0.30)
    plt.legend(bbox_to_anchor=(0, -0.09, 1, -0.09), loc="upper left",
               mode="expand", borderaxespad=0, ncol=1)
    # plt.legend(loc="lower center")
