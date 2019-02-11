"""Copy a balanced set of files from a source image collection
Colin Dietrich 2019
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('VGG16 Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('VGG16 Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          percent=False,
                          title=None,
                          show_color_bar=True,
                          figsize=(6, 6),
                          dpi=50,
                          cmap=plt.cm.Blues,
                          axis_color='black',
                          save=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Modified from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Parameters
    ----------
    cm : confusion matrix object from scikit-learn
    normalize : bool, normalize values
    percent : bool, convert values to percentage
    title : str, title of plot
    show_color_bar : bool, show color bar on right side of plot
    figsize : tuple of (height, width) in inches, size of output plot
    dpi : int, dots per inch of output plot
    cmap : matplotlib colormap for plot
    axis_color : str, color to make outside ticks and labels
    save : bool, save plot to transparent PNG
    """

    n_str = ""

    if (normalize is True) & (percent is True):
        cm = (cm.astype('float') * 100 / cm.sum(axis=1)[:, np.newaxis]).astype(int)
        fmt = 'd'
    elif normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        fmt = '.2f'
    else:
        n_str = " with n={}".format(cm.sum())
        fmt = 'd'
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title + n_str)

    if show_color_bar:
        color_bar = plt.colorbar()
        color_bar.ax.tick_params(axis='y', colors=axis_color)
        color_bar.ax.yaxis.label.set_color(axis_color)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        n = cm[i, j]
        s = ''
        if percent:
            s = ' %'
        plt.text(j, i, format(n, fmt) + s,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax = plt.gca()

    ax.xaxis.label.set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.tick_params(axis='y', colors=axis_color)

    ax.title.set_color(axis_color)

    plt.tight_layout()
    if save:
        plt.savefig('confusion_matrix.png', bbox_inches='tight', transparent=True)
    plt.show()
