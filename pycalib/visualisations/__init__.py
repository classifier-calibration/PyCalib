import numpy as np
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import label_binarize

from statsmodels.stats.proportion import proportion_confint

from matplotlib import gridspec

from pycalib.utils import (df_normalise, multiindex_to_strings,
                          get_binned_scores)


def plot_reliability_diagram_precomputed(avg_true, avg_pred,
                                         legend=None,
                                         class_names=None,
                                         fig=None,
                                         fmt='s-',
                                         show_correction=False,
                                         show_gaps=False,
                                         color_list=None,
                                         color_gaps='lightcoral'):
    """ Plots the reliability diagram for precomputed averaged scores and labels

    NOTE: This function is currently a copy from plot_reliability_diagram and
    modified to accept average scores and true proportions. In the future both
    functions may be merged or share common private functions.
    Parameters
    ==========
    avg_true : matrix (n_bins, n_classes) or list of matrices
        True proportions per class.
    avg_pred : matrix (n_bins, n_classes) or list of matrices
        Output probability scores for one or several methods.
    legend : list of strings or None
        Text to use for the legend.
    bins : int or list of floats
        Number of bins to create in the scores' space, or list of bin
        boundaries.
    class_names : list of strings or None
        Name of each class, if None it will assign integer numbers starting
        with 1.
    fig : matplotlib.pyplot.Figure or None
        Figure to use for the plots, if None a new figure is created.
    show_counts : boolean
        If True shows the number of samples of each bin in its corresponding
        line marker.
    interval_method : string (default: 'beta')
        Method to estimate the confidence interval which uses the function
        proportion_confint from statsmodels.stats.proportion
    fmt : string (default: 's-')
        Format of the lines following the matplotlib.pyplot.plot standard.
    show_correction : boolean
        If True shows an arrow for each bin indicating the necessary correction
        to the average scores in order to be perfectly calibrated.
    show_gaps : boolean
        If True shows the gap between the average predictions and the true
        proportion of positive samples.
    sample_proportion : float in the interval [0, 1] (default 0)
        If bigger than 0, it shows the labels of the specified proportion of
        samples.
    color_list : list of strings or None
        List of string colors indicating the color of each method.
    color_gaps : string
        Color of the gaps (if shown).

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    """
    if isinstance(avg_true, list):
        avg_true_list = avg_true
    else:
        avg_true_list = [avg_true, ]
    if isinstance(avg_pred, list):
        avg_pred_list = avg_pred
    else:
        avg_pred_list = [avg_pred, ]

    n_classes = avg_true_list[0].shape[1]
    n_scores = len(avg_true_list)

    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if class_names is None:
        class_names = [str(i+1) for i in range(n_classes)]

    if n_classes == 2:
        avg_pred_list = [pred[:, 1].reshape(-1, 1) for pred in avg_pred_list]
        class_names = [class_names[1], ]

    n_columns = n_classes if n_classes != 2 else 1

    if fig is None:
        fig = plt.figure(figsize=(n_columns*4, 4))

    spec = gridspec.GridSpec(ncols=n_columns, nrows=1, wspace=0.02,
                             hspace=0.04, left=0.15)

    for i in range(n_columns):
        ax1 = fig.add_subplot(spec[i])
        # Perfect calibration
        ax1.plot([0, 1], [0, 1], "--", color='lightgrey',
                 zorder=0)

        for j in range(n_scores):
            # bin_total = bin_total_list[j][:, i]
            pred_sort_idx = np.argsort(avg_pred_list[j][:, i])
            avg_true = avg_true_list[j][pred_sort_idx, i]
            avg_pred = avg_pred_list[j][pred_sort_idx, i]

            name = legend[j] if legend else None
            ax1.plot(avg_pred, avg_true, fmt, label=name, color=color_list[j])

            if show_correction:
                for ap, at in zip(avg_pred, avg_true):
                    ax1.arrow(ap, at, at - ap, 0, color=color_gaps,
                              head_width=0.02, length_includes_head=True,
                              width=0.01)

            if show_gaps:
                for ap, at in zip(avg_pred, avg_true):
                    ygaps = avg_pred - avg_true
                    ygaps = np.vstack((np.zeros_like(ygaps), ygaps))
                    ax1.errorbar(avg_pred, avg_true, yerr=ygaps, fmt=" ",
                                 color=color_gaps, lw=4, capsize=5, capthick=1,
                                 zorder=10)

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('Average score (Class {})'.format(class_names[i]))
        if i == 0:
            ax1.set_ylabel('Fraction of positives')
        else:
            ax1.set_yticklabels([])
            nbins = len(ax1.get_xticklabels())
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=nbins,
                                                    prune='lower'))
        ax1.grid(True)
        ax1.set_axisbelow(True)

    if legend is not None:
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center',
                   bbox_to_anchor=(0, 0, 1, 1),
                   bbox_transform=fig.transFigure, ncol=6)

    fig.align_labels()
    return fig


def plot_reliability_diagram(labels, scores, legend=None,
                             show_histogram=True,
                             bins=10, class_names=None, fig=None,
                             show_counts=False, errorbar_interval=None,
                             interval_method='beta', fmt='s-',
                             show_correction=False,
                             show_gaps=False,
                             sample_proportion=0,
                             hist_per_class=False,
                             color_list=None,
                             show_bars=False,
                             invert_histogram=False,
                             color_gaps='lightcoral',
                             confidence=False):
    """ Plots the reliability diagram of the given scores and true labels

    Parameters
    ==========
    labels : array (n_samples, )
        Labels indicating the true class.
    scores : matrix (n_samples, n_classes) or list of matrices
        Output probability scores for one or several methods.
    legend : list of strings or None
        Text to use for the legend.
    show_histogram : boolean
        If True, it generates an additional figure showing the number of
        samples in each bin.
    bins : int or list of floats
        Number of bins to create in the scores' space, or list of bin
        boundaries.
    class_names : list of strings or None
        Name of each class, if None it will assign integer numbers starting
        with 1.
    fig : matplotlib.pyplot.Figure or None
        Figure to use for the plots, if None a new figure is created.
    show_counts : boolean
        If True shows the number of samples of each bin in its corresponding
        line marker.
    errorbar_interval : float or None
        If a float between 0 and 1 is passed, it shows an errorbar
        corresponding to a confidence interval containing the specified
        percentile of the data.
    interval_method : string (default: 'beta')
        Method to estimate the confidence interval which uses the function
        proportion_confint from statsmodels.stats.proportion
    fmt : string (default: 's-')
        Format of the lines following the matplotlib.pyplot.plot standard.
    show_correction : boolean
        If True shows an arrow for each bin indicating the necessary correction
        to the average scores in order to be perfectly calibrated.
    show_gaps : boolean
        If True shows the gap between the average predictions and the true
        proportion of positive samples.
    sample_proportion : float in the interval [0, 1] (default 0)
        If bigger than 0, it shows the labels of the specified proportion of
        samples.
    hist_per_class : boolean
        If True shows one histogram of the bins per class.
    color_list : list of strings or None
        List of string colors indicating the color of each method.
    show_bars : boolean
        If True shows bars instead of lines.
    invert_histogram : boolean
        If True shows the histogram with the zero on top and highest number of
        bin samples at the bottom.
    color_gaps : string
        Color of the gaps (if shown).
    confidence : boolean
        If True shows only the confidence reliability diagram.

    Regurns
    =======
    fig : matplotlib.pyplot.figure
        Figure with the reliability diagram
    """
    if isinstance(scores, list):
        scores_list = scores
    else:
        scores_list = [scores, ]
    n_scores = len(scores_list)
    if color_list is None:
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']

    classes = np.arange(scores_list[0].shape[1])
    n_classes = len(classes)
    labels = label_binarize(labels, classes=classes)

    labels_list = []
    if confidence:
        labels_idx = np.argmax(labels, axis=1)
        new_scores_list = []
        for score in scores_list:
            # TODO: randomize selection when there are several winning classes
            conf_idx = np.argmax(score, axis=1)
            winning_score = np.max(score, axis=1)
            new_scores_list.append(np.vstack([1 - winning_score,
                                              winning_score]).T)
            labels_list.append((conf_idx.flatten()
                               == labels_idx.flatten()).astype(int))
            labels_list[-1] = label_binarize(labels_list[-1], classes=[0, 1])
        scores_list = new_scores_list
        n_classes = 2
        class_names = ['Non winning', 'winning']
        n_columns = 1
    else:
        n_columns = labels.shape[1]

    if class_names is None:
        class_names = [str(i+1) for i in range(n_classes)]

    if n_classes == 2:
        scores_list = [score[:, 1].reshape(-1, 1) for score in scores_list]
        class_names = [class_names[1], ]

    if fig is None:
        fig = plt.figure(figsize=(n_columns*4, 4))

    if show_histogram:
        spec = gridspec.GridSpec(ncols=n_columns, nrows=2,
                                 height_ratios=[5, 1],
                                 wspace=0.02,
                                 hspace=0.04,
                                 left=0.15)
    else:
        spec = gridspec.GridSpec(ncols=n_columns, nrows=1,
                                 hspace=0.04, left=0.15)

    if isinstance(bins, int):
        n_bins = bins
        bins = np.linspace(0, 1 + 1e-8, n_bins + 1)
    elif isinstance(bins, list) or isinstance(bins, np.ndarray):
        n_bins = len(bins) - 1
        bins = np.array(bins)
        if bins[0] == 0.0:
            bins[0] = 0 - 1e-8
        if bins[-1] == 1.0:
            bins[-1] = 1 + 1e-8

    for i in range(n_columns):
        ax1 = fig.add_subplot(spec[i])
        # Perfect calibration
        ax1.plot([0, 1], [0, 1], "--", color='lightgrey',
                 zorder=0)
        for j, score in enumerate(scores_list):
            if labels_list:
                labels = labels_list[j]

            avg_true, avg_pred, bin_true, bin_total = get_binned_scores(
                labels[:, i], score[:, i], bins=bins)
            zero_idx = bin_total == 0

            name = legend[j] if legend else None
            if show_bars:
                ax1.bar(x=bins[:-1][~zero_idx], height=avg_true[~zero_idx],
                        align='edge', width=(bins[1:] - bins[:-1])[~zero_idx],
                        edgecolor='black', color=color_list[j])
            else:
                if errorbar_interval is None:
                    ax1.plot(avg_pred, avg_true, fmt, label=name,
                             color=color_list[j])
                else:
                    nozero_intervals = proportion_confint(
                        count=bin_true[~zero_idx], nobs=bin_total[~zero_idx],
                        alpha=1-errorbar_interval,
                        method=interval_method)
                    nozero_intervals = np.array(nozero_intervals)

                    intervals = np.empty((2, bin_total.shape[0]))
                    intervals.fill(np.nan)
                    intervals[:, ~zero_idx] = nozero_intervals

                    yerr = intervals - avg_true
                    yerr = np.abs(yerr)
                    ax1.errorbar(avg_pred, avg_true, yerr=yerr, label=name,
                                 fmt=fmt, color=color_list[j])  # markersize=5)

            if show_counts:
                for ap, at, count in zip(avg_pred, avg_true, bin_total):
                    if np.isfinite(ap) and np.isfinite(at):
                        ax1.text(ap, at, str(count), fontsize=6,
                                 ha='center', va='center', zorder=11,
                                 bbox=dict(boxstyle='square,pad=0.3',
                                           fc='white', ec=color_list[j]))

            if show_correction:
                for ap, at in zip(avg_pred, avg_true):
                    ax1.arrow(ap, at, at - ap, 0, color=color_gaps,
                              head_width=0.02, length_includes_head=True,
                              width=0.01)

            if show_gaps:
                for ap, at in zip(avg_pred, avg_true):
                    ygaps = avg_pred - avg_true
                    ygaps = np.vstack((np.zeros_like(ygaps), ygaps))
                    ax1.errorbar(avg_pred, avg_true, yerr=ygaps, fmt=" ",
                                 color=color_gaps, lw=4, capsize=5, capthick=1,
                                 zorder=10)

            if sample_proportion > 0:
                idx = np.random.choice(labels.shape[0],
                                       int(sample_proportion*labels.shape[0]))
                ax1.scatter(score[idx, i], labels[idx, i], marker='|', s=100,
                            alpha=0.2, color=color_list[j])

        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        # ax1.set_title('Class {}'.format(class_names[i]))
        if not show_histogram:
            ax1.set_xlabel('Average score (Class {})'.format(
                class_names[i]))
        if i == 0:
            ax1.set_ylabel('Fraction of positives')
        else:
            ax1.set_yticklabels([])
        ax1.grid(True)
        ax1.set_axisbelow(True)

        if show_histogram:
            ax2 = fig.add_subplot(spec[n_columns + i],
                                  label='{}'.format(i))
            for j, score in enumerate(scores_list):
                ax1.set_xticklabels([])
                # lines = ax1.get_lines()
                # ax2.set_xticklabels([])

                name = legend[j] if legend else None
                if hist_per_class:
                    for c in [0, 1]:
                        linestyle = ('dotted', 'dashed')[c]
                        ax2.hist(score[labels[:, i] == c, i], range=(0, 1),
                                 bins=bins, label=name,
                                 histtype="step",
                                 lw=1, linestyle=linestyle,
                                 color=color_list[j],
                                 edgecolor='black')
                else:
                    if n_scores > 1:
                        kwargs = {'histtype': 'step',
                                  'edgecolor': color_list[j]}
                    else:
                        kwargs = {'histtype': 'bar',
                                  'edgecolor': 'black',
                                  'color': color_list[j]}
                    ax2.hist(score[:, i], range=(0, 1), bins=bins, label=name,
                             lw=1, **kwargs)
                ax2.set_xlim([0, 1])
                ax2.set_xlabel('Average score (Class {})'.format(
                    class_names[i]))
                ax2.yaxis.set_major_locator(MaxNLocator(integer=True,
                                                        prune='upper',
                                                        nbins=3))
            if i == 0:
                ax2.set_ylabel('Count')
                ytickloc = ax2.get_yticks()
                ax2.yaxis.set_major_locator(mticker.FixedLocator(ytickloc))
                yticklabels = ['{:0.0f}'.format(value) for value in
                               ytickloc]
                ax2.set_yticklabels(labels=yticklabels,
                                    fontdict=dict(verticalalignment='top'))
            else:
                ax2.set_yticklabels([])
                nbins = len(ax2.get_xticklabels())
                ax2.xaxis.set_major_locator(MaxNLocator(nbins=nbins,
                                                        prune='lower'))
            ax2.grid(True, which='both')
            ax2.set_axisbelow(True)
            if invert_histogram:
                ylim = ax2.get_ylim()
                ax2.set_ylim(reversed(ylim))

    if legend is not None:
        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='upper center',
                   bbox_to_anchor=(0, 0, 1, 1),
                   bbox_transform=fig.transFigure, ncol=6)

    fig.align_labels()
    return fig


def plot_binary_reliability_diagram_gaps(y_true, p_pred, n_bins=15, title=None,
                                         fig=None, ax=None, legend=False,
                                         color_gaps='lightcoral'):
    """Plot binary reliability diagram gaps

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
    """
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
            # First bin includes lower bound
            bin_indices = np.where(np.logical_and(
                p_pred >= center - bin_size/2,
                p_pred <= center + bin_size/2))
        else:
            bin_indices = np.where(np.logical_and(p_pred > center - bin_size/2,
                                                  p_pred <= center +
                                                  bin_size/2))
        if len(bin_indices[0]) == 0:
            true_proportion[i] = np.nan
            pred_mean[i] = np.nan
        else:
            true_proportion[i] = np.mean(y_true[bin_indices])
            pred_mean[i] = np.nanmean(p_pred[bin_indices])

    not_nan = np.isfinite(true_proportion - centers)
    ax.bar(centers, true_proportion, width=bin_size, edgecolor="black",
           # color="blue", label='True class prop.')
           color="cornflowerblue", label='True class prop.')
    ax.bar(pred_mean[not_nan], (true_proportion - pred_mean)[not_nan],
           bottom=pred_mean[not_nan], width=0.01,
           edgecolor=color_gaps,
           color=color_gaps,
           label='Gap pred. mean', align='center')

    if legend:
        ax.legend()

    ax.plot([0, 1], [0, 1], linestyle="--", color='grey')
    ax.set_xlim([0, 1])
    ax.set_xlabel('Predicted probability')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Proportion of positives')
    ax.grid(True)
    ax.set_axisbelow(True)

    return fig, ax


def plot_multiclass_reliability_diagram_gaps(y_true, p_pred, fig=None, ax=None,
                                             per_class=True, legend=False,
                                             **kwargs):

    if len(y_true.shape) < 2 or y_true.shape[1] == 1:
        ohe = OneHotEncoder(categories='auto')
        ohe.fit(y_true.reshape(-1, 1))
        y_true = ohe.transform(y_true.reshape(-1, 1))

    if per_class:
        n_classes = y_true.shape[1]
        if fig is None and ax is None:
            fig = plt.figure(figsize=((n_classes-1)*4, 4))
        if ax is None:
            ax = [fig.add_subplot(1, n_classes, i+1) for i in range(n_classes)]
        for i in range(n_classes):
            if i == 0 and legend:
                sub_legend = True
            else:
                sub_legend = False
            plot_binary_reliability_diagram_gaps(y_true[:, i], p_pred[:, i],
                                                 title='$C_{}$'.format(i+1),
                                                 fig=fig, ax=ax[i],
                                                 legend=sub_legend,
                                                 **kwargs)
            if i > 0:
                ax[i].set_ylabel('')
            ax[i].set_xlabel('Predicted probability')
    else:
        if fig is None and ax is None:
            fig = plt.figure()
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


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues,
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
    return fig


def plot_individual_pdfs(class_dist, x_grid=None, y_grid=None,
                         grid_levels=200, fig=None, title=None,
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

    for i, (p, d) in enumerate(zip(class_dist.priors,
                                   class_dist.distributions)):
        z = d.pdf(np.vstack([xx.flatten(), yy.flatten()]).T)

        ax = fig.add_subplot(1, len(class_dist.distributions), i+1)
        if title is None:
            ax.set_title('$P(Y={})={:.2f}$\n{}'.format(i+1, p, str(d)),
                         loc='left')
        else:
            ax.set_title(title[i])
        contour = ax.contourf(xx, yy, z.reshape(grid_levels, grid_levels),
                              cmap=cmaps[i])
        if grid:
            ax.grid()
        fig.colorbar(contour)

    return fig


def plot_critical_difference(avranks, num_datasets, names, title=None,
                             test='bonferroni-dunn'):
    """
        test: string in ['nemenyi', 'bonferroni-dunn']
         - nemenyi two-tailed test (up to 20 methods)
         - bonferroni-dunn one-tailed test (only up to 10 methods)

    """
    # Critical difference plot
    import Orange

    if len(avranks) > 10:
        print('Forcing Nemenyi Critical difference')
        test = 'nemenyi'
    cd = Orange.evaluation.compute_CD(avranks, num_datasets, alpha='0.05',
                                      test=test)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6,
                                  textspace=1.5)
    fig = plt.gcf()
    fig.suptitle(title, horizontalalignment='left')
    return fig


def plot_df_to_heatmap(df, title=None, figsize=None, annotate=True,
                       normalise_columns=False, normalise_rows=False,
                       cmap=None):
    """ Exports a heatmap of the given pandas DataFrame

    Parameters
    ----------
    df:     pandas.DataFrame
        It should be a matrix, it can have multiple index and these will be
        flattened.

    title: string
        Title of the figure

    figsize:    tuple of ints (x, y)
        Figure size in inches

    annotate:   bool
        If true, adds numbers inside each box
    """
    if normalise_columns:
        df = df_normalise(df, columns=True)
    if normalise_rows:
        df = df_normalise(df, columns=False)

    yticklabels = multiindex_to_strings(df.index)
    xticklabels = multiindex_to_strings(df.columns)
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        point_inch_ratio = 72.
        n_rows = df.shape[0]
        font_size_pt = plt.rcParams['font.size']
        xlabel_space_pt = max([len(xlabel) for xlabel in xticklabels])
        fig_height_in = (((xlabel_space_pt + n_rows) * (font_size_pt + 3))
                         / point_inch_ratio)

        n_cols = df.shape[1]
        fig_width_in = df.shape[1]+4
        ylabel_space_pt = max([len(ylabel) for ylabel in yticklabels])
        fig_width_in = ((ylabel_space_pt + (n_cols * 3) + 5)
                        * (font_size_pt + 3)) / point_inch_ratio
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    cax = ax.pcolor(df, cmap=cmap)
    fig.colorbar(cax)
    ax.set_yticks(np.arange(0.5, len(df.index), 1))
    ax.set_yticklabels(yticklabels)
    ax.set_xticks(np.arange(0.5, len(df.columns), 1))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    middle_value = (df.max().max() + df.min().min())/2.0
    if annotate:
        for y in range(df.shape[0]):
            for x in range(df.shape[1]):
                color = 'white' if middle_value > df.values[y, x] else 'black'
                plt.text(x + 0.5, y + 0.5, '%.2f' % df.values[y, x],
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=color
                         )
    return fig
