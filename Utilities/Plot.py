import math
import time
import pandas
import matplotlib.pyplot as plt
import matplotlib
from Utilities.Distance import levenshtein


def plot_losses(loss: list, x_label: str, y_label: str, folder: str = "Result", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_pie_of_levenshtein(df: pandas.DataFrame, title: str):
    counts_dict = dict()

    for _, row in df.iterrows():
        noised = str(row['Noised'])
        correct = str(row['Correct'])

        distance = levenshtein(noised, correct)

        if distance in counts_dict.keys():
            counts_dict[distance] += 1
        else:
            counts_dict[distance] = 1

    distances = counts_dict.keys()
    counts = counts_dict.values()

    fig, ax = plt.subplots(figsize=(len(counts), 3),
                           subplot_kw=dict(aspect="equal"))

    def func(pct, allvals):
        return "{:.1f}%".format(pct)

    wedges, texts, autotexts = ax.pie(counts, autopct=lambda pct: func(pct, counts),
                                      textprops=dict(color="w"))

    ax.legend(wedges, distances,
              title="Distance",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title(title)

    plt.show()


def show_scatter_plot(x: list, y: list, x_label: str, y_label: str):
    plt.scatter(x, y, color='g')
    yint = range(min(y), max(y) + 1)
    xint = range(min(x), max(x) + 1)
    matplotlib.pyplot.yticks(yint)
    matplotlib.pyplot.xticks(xint)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def show_box_plot(df: pandas.DataFrame, column: str, by: str, save_pth: str):
    df.boxplot(column=column, by=by)
    plt.savefig(save_pth)
