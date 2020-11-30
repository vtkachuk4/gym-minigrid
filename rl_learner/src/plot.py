import matplotlib.pyplot as plt


def plot_progress(x_axis, *argv):
    alpha = 0.5
    linewidth = 2
    iters = len(argv) // 2

    plt.rcParams["figure.figsize"] = (13, 5)
    for i in range(iters):
        plt.plot(x_axis, argv[i * 2 + 0], label=argv[i * 2 + 1], alpha=alpha,
                 linewidth=linewidth)

    plt.xlabel('Episode number')
    plt.ylabel('Statistic')
    plt.subplots_adjust(bottom=0.20)
    plt.legend(bbox_to_anchor=(0, -0.09, 1, -0.09), loc="upper left",
               mode="expand", borderaxespad=0, ncol=2)
