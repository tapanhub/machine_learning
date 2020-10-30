import matplotlib.pyplot as plt


def xyplot(x, y, x_label='x', y_label='y', line_label='line'):
    #fig, ax = plt.subplot()
    with plt.style.context('seaborn-darkgrid'):
        fig, ax = plt.subplots(figsize=(4.5, 3), dpi=100)
        line = ax.plot(x, y, label=line_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        legend = ax.legend(loc='upper right')
    plt.show()
