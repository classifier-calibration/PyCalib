import numpy as np
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


def plot_reliability_diagram(labels, scores, legend=None, histogram=True,
                             n_bins=10, class_names=None, fig=None):
    '''
    Parameters
    ==========
    labels : array (n_samples, )
        Labels indicating the ground class
    scores : list of matrices [(n_samples, n_classes)]
        Output probability scores for every method
    legend : list of strings
        Text to use for the legend
    n_bins : int
        Number of bins to create in the scores' space
    histogram : boolean
        If True, it generates an additional figure showing the number of
        samples in each bin.

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    fig2 : matplotlib.pyplot.figure
        Only if histogram == True
    '''
    classes = np.unique(labels)
    n_classes = len(classes)
    labels = label_binarize(labels, classes=classes)

    if fig is None:
        fig = plt.figure()

    if n_classes == 2:
        scores = [score[:, 1].reshape(-1, 1) for score in scores]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    n_columns = labels.shape[1]
    for i in range(n_columns):
        ax1 = fig.add_subplot(1, n_columns, i+1)
        ax1.set_title('Class {}'.format(class_names[i]))
        for score, name in zip(scores, legend):
            fraction_of_positives, mean_predicted_value = calibration_curve(labels[:, i], score[:, i],
                                                                            n_bins=n_bins)
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                             label=name)
        ax1.plot([0, 1], [0, 1], "k--")
        ax1.legend()
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Mean predicted value')
        ax1.set_ylabel('Fraction of positives')
        ax1.grid()

    if histogram:
        if n_classes == 2:
            fig2 = plt.figure(figsize=(5, 5))
        else:
            fig2 = plt.figure(figsize=(15, 5))

        for i in range(n_columns):
            ax = fig2.add_subplot(1, n_columns, i+1)
            for score, name in zip(scores, legend):
                ax.hist(score[:, i], range=(0, 1), bins=n_bins, label=name,
                         histtype="step", lw=2)
                ax.legend()
                ax.set_xlim([0, 1])
                ax.set_xlabel('Mean predicted value')
                ax.set_ylabel('Number of samples in bin')
                ax.grid()
        return fig, fig2
    return fig


def plot_binary_reliability_diagram_gaps(y_true, p_pred, n_bins=15, title=None,
                                         fig=None, ax=None, legend=True):
    '''Plot binary reliability diagram gaps

    Parameters
    ==========
    y_true : np.array shape (n_samples, 2) or (n_samples, )
        Labels corresponding to the scores as a binary indicator matrix or as a
        vector of integers indicating the class.
    p_pred : binary matrix shape (n_samples, 2) or (n_samples, )
        Output probability scores for each class as a matrix, or for the
        positive class
    n_bins : integer
        Number of bins to divide the scores
    title : string
        Title for the plot
    fig : matplotlib.pyplot.figure
        Plots the axis in the given figure
    ax : matplotlib.pyplot.Axis
        Axis where to draw the plot
    legend : boolean
        If True the function will draw a legend

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    '''
    if fig is None and ax is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    if (len(y_true.shape) == 2) and (y_true.shape[1] == 2):
        y_true = y_true[:, 1]
    if (len(y_true.shape) == 2) and (y_true.shape[1] > 2):
        raise ValueError('y_true wrong dimensions {}'.format(y_true.shape))

    if (len(p_pred.shape) == 2) and (p_pred.shape[1] == 2):
        p_pred = p_pred[:, 1]
    if (len(p_pred.shape) == 2) and (p_pred.shape[1] > 2):
        raise ValueError('p_pred wrong dimensions {}'.format(p_pred.shape))

    bin_size = 1.0/n_bins
    centers = np.linspace(bin_size/2.0, 1.0 - bin_size/2.0, n_bins)
    true_proportion = np.zeros(n_bins)
    pred_mean = np.zeros(n_bins)
    for i, center in enumerate(centers):
        if i == 0:
            # First bin include lower bound
            bin_indices = np.where(np.logical_and(p_pred >= center - bin_size/2,
                                                  p_pred <= center +
                                                  bin_size/2))
        else:
            bin_indices = np.where(np.logical_and(p_pred > center - bin_size/2,
                                                  p_pred <= center +
                                                  bin_size/2))
        true_proportion[i] = np.mean(y_true[bin_indices])
        pred_mean[i] = np.nanmean(p_pred[bin_indices])

    not_nan = np.isfinite(true_proportion - centers)
    ax.bar(centers, true_proportion, width=bin_size, edgecolor="black",
           color="blue", label='True class prop.')
    ax.bar(pred_mean[not_nan], (true_proportion - pred_mean)[not_nan],
           bottom=pred_mean[not_nan], width=bin_size/4.0, edgecolor="red",
           color="#ffc8c6",
           label='Gap pred. mean')
    ax.scatter(pred_mean[not_nan], true_proportion[not_nan], color='red',
               marker="+", zorder=10)

    if legend:
        ax.legend()

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_xlabel('Predicted probability')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Proportion of positives')
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.tight_layout()

    return fig, ax


def plot_multiclass_reliability_diagram_gaps(y_true, p_pred, fig=None, ax=None,
                                             per_class=True, **kwargs):
    if fig is None and ax is None:
        fig = plt.figure()

    if len(y_true.shape) < 2 or y_true.shape[1] == 1:
        ohe = OneHotEncoder(categories='auto')
        ohe.fit(y_true.reshape(-1, 1))
        y_true = ohe.transform(y_true.reshape(-1,1))

    if per_class:
        n_classes = y_true.shape[1]
        if ax is None:
            ax = [fig.add_subplot(1, n_classes, i+1) for i in range(n_classes)]
        for i in range(n_classes):
            plot_binary_reliability_diagram_gaps(y_true[:,i], p_pred[:,i],
                                                title=r'$C_{}$'.format(i+1),
                                                fig=fig, ax=ax[i], **kwargs)
            if i > 0:
                ax[i].set_ylabel('')
    else:
        mask = p_pred.argmax(axis=1)
        indices = np.arange(p_pred.shape[0])
        y_true = y_true[indices, mask].T
        p_pred = p_pred[indices, mask].T
        ax = fig.add_subplot(1, 1, 1)
        plot_binary_reliability_diagram_gaps(y_true, p_pred,
                                             title=r'$C_1$',
                                             fig=fig, ax=ax, **kwargs)
        ax.set_title('')

    return fig

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fig=None, ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return fig

def plot_weight_matrix(weights, bias, classes, title='Weight matrix',
                       cmap=plt.cm.Greens, fig=None, ax=None, **kwargs):
    """
    This function prints and plots the weight matrix.
    """
    if fig is None:
        fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot(111)

    if title is not None:
        ax.set_title(title)

    matrix = np.hstack((weights, bias.reshape(-1, 1)))

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap, **kwargs)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    tick_marks = np.arange(len(classes))
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xticks(np.append(tick_marks, len(classes)))
    ax.set_xticklabels(np.append(classes, 'c'))

    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    ax.set_ylabel('Class')
    fig.tight_layout()
    return fig


def plot_individual_pdfs(class_dist, x_grid=None, y_grid=None,
                         grid_levels = 200, fig=None, title=None,
                         cmaps=None, grid=True):
    if fig is None:
        fig = plt.figure()

    if x_grid is None:
        x_grid = np.linspace(-8, 8, grid_levels)
    else:
        grid_levels = len(x_grid)

    if y_grid is None:
        y_grid = np.linspace(-8, 8, grid_levels)

    xx, yy = np.meshgrid(x_grid, y_grid)

    if cmaps is None:
        cmaps = [None]*len(class_dist.priors)

    for i, (p, d) in enumerate(zip(class_dist.priors, class_dist.distributions)):
        z = d.pdf(np.vstack([xx.flatten(), yy.flatten()]).T)

        ax = fig.add_subplot(1, len(class_dist.distributions), i+1)
        if title is None:
            ax.set_title('$P(Y={})={:.2f}$\n{}'.format(i+1, p, str(d)), loc='left')
        else:
            ax.set_title(title[i])
        contour = ax.contourf(xx, yy, z.reshape(grid_levels,grid_levels),
                              cmap=cmaps[i])
        if grid:
            ax.grid()
        fig.colorbar(contour)

    return fig
