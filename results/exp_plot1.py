import matplotlib.pyplot as plt
import numpy as np

N = 10


names = ['D-W','D-A','A-W','A-D','W-A','W-D']
celists = [
[[11, 30, 21, 40, 11, 25, 22, 24, 23, 17], [9, 0, 6, 0, 0, 0, 8, 0, 0, 4]],
[[93, 73, 12, 65, 52, 66, 82, 77, 47, 29], [7, 25, 86, 25, 23, 34, 17, 22, 49, 35]],
[[2, 25, 20, 34, 10, 24, 19, 13, 20, 19], [18, 5, 7, 6, 1, 1, 11, 11, 3, 2]],
[[4, 12, 1, 8, 6, 18, 19, 17, 17, 10], [11, 11, 17, 2, 1, 0, 7, 4, 5, 5]],
[[79, 85, 25, 73, 54, 72, 80, 63, 27, 15], [21, 13, 73, 17, 21, 28, 19, 36, 69, 49]],
[[15, 22, 18, 10, 7, 18, 26, 19, 22, 15], [0, 1, 0, 0, 0, 0, 0, 2, 0, 0]]
]



for index, ce in enumerate(celists):
    correct, error = ce

    CorrectRate = []
    ErrorRate = []
    for i,j in zip(correct, error):
        total = i+j
        CorrectRate.append(i/total*100)
        ErrorRate.append(j/total*100)

    # CorrectRate = set(CorrectRate)
    # ErrorRate = set(ErrorRate)
    # print(CorrectRate)
    # print(ErrorRate)
    ind = np.arange(10, N+10)    # the x locations for the groups
    width = 0.8       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, 5))

    p1 = ax.bar(ind, CorrectRate, width, label='CorrectRate', color='blue', alpha=0.4)
    p2 = ax.bar(ind, ErrorRate, width,
                bottom=CorrectRate, label='ErrorRate', color='red', alpha=0.4)

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Unknown category')
    ax.set_title('Fineness estimation')
    ax.set_xticks(ind)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend(loc=4)

    # Label with label_type 'center' instead of the default 'edge'
    ax.bar_label(p1, label_type='center', fmt='%d%%')
    # ax.bar_label(p2, label_type='center')
    fig.tight_layout()
    print(names[index], CorrectRate)
    # plt.savefig(f"results/{names[index]}-Fineness.pdf", dpi=150)
    # plt.show()