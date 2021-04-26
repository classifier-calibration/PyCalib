import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize
from scipy.stats import percentileofscore


def accuracy(y_true, y_pred):
    """Classification accuracy score

    Accuracy for binary and multiclass classification problems. Consists on the
    proportion of correct estimations assuming the maximum class probability of
    each score as the estimated class.

    Parameters
    ----------
    y_true : label indicator matrix (n_samples, n_classes)
        True labels.
        # TODO Add option to pass array with shape (n_samples, )

    y_pred : matrix (n_samples, n_classes)
        Predicted scores.

    Returns
    -------
    score : float
        Proportion of correct predictions as a value between 0 and 1.

    Examples
    --------
    >>> from pycalib.metrics import accuracy
    >>> Y = np.array([[0, 1], [0, 1]])
    >>> S = np.array([[0.1, 0.9], [0.6, 0.4]])
    >>> accuracy(Y, S)
    0.5
    >>> Y = np.array([[0, 1], [0, 1]])
    >>> S = np.array([[0.1, 0.9], [0, 1]])
    >>> accuracy(Y, S)
    1.0
    """
    predictions = np.argmax(y_pred, axis=1)
    y = np.argmax(y_true, axis=1)
    return np.mean(predictions == y)


def cross_entropy(y_true, y_pred):
    """Cross-entropy score

    Computes the cross-entropy (a.k.a. log-loss) for binary and
    multiclass classification scores.

    Parameters
    ----------
    y_true : label indicator matrix (n_samples, n_classes)
        True labels.
        # TODO Add option to pass array with shape (n_samples, )

    y_pred : matrix (n_samples, n_classes)
        Predicted scores.

    Returns
    -------
    score : float

    Examples
    --------
    >>> from pycalib.metrics import cross_entropy
    >>> Y = np.array([[0, 1], [0, 1]])
    >>> S = np.array([[0.1, 0.9], [0.6, 0.4]])
    >>> cross_entropy(Y, S)
    0.5108256237659906
    """
    return log_loss(y_true, y_pred)


def brier_score(y_true, y_pred):
    """Brier score

    Computes the Brier score between the true labels and the estimated
    probabilities. This corresponds to the Mean Squared Error between the
    estimations and the true labels.

    Parameters
    ----------
    y_true : label indicator matrix (n_samples, n_classes)
        True labels.
        # TODO Add option to pass array with shape (n_samples, )

    y_pred : matrix (n_samples, n_classes)
        Predicted scores.

    Returns
    -------
    score : float
        Positive value between 0 and 1.

    Examples
    --------
    >>> from pycalib.metrics import cross_entropy
    >>> Y = np.array([[0, 1], [0, 1]])
    >>> S = np.array([[0.1, 0.9], [0.6, 0.4]])
    >>> brier_score(Y, S)
    0.185
    """
    # TODO Consider using the following code instead
    # np.mean(np.abs(S - Y)**2)
    return mean_squared_error(y_true, y_pred)


def conf_ECE(y_true, probs, bins=15):
    """
    Calculate ECE score based on model output probabilities and true labels

    Parameters
    ==========
    y_true:
        - a list containing the actual class labels
        - ndarray shape (n_samples) with a list containing actual class
          labels
        - ndarray shape (n_samples, n_classes) with largest value in
          each row for the correct column class.
    probs:
        a list containing probabilities for all the classes with a shape of
        (samples, classes)
    bins: (int)
        - into how many bins are probabilities divided (default = 15)

    Returns
    =======
    ece : float
        expected calibration error
    """
    return ECE(y_true, probs, normalize=False, bins=bins, ece_full=False)


def ECE(y_true, probs, normalize=False, bins=15, ece_full=True):
    """
    Calculate ECE score based on model output probabilities and true labels

    Parameters
    ==========
    y_true : list
        a list containing the actual class labels
        ndarray shape (n_samples) with a list containing actual class
          labels
        ndarray shape (n_samples, n_classes) with largest value in
          each row for the correct column class.
    probs : list
        a list containing probabilities for all the classes with a shape of
        (samples, classes)
    normalize: (bool)
        in case of 1-vs-K calibration, the probabilities need to be
        normalized. (default = False)
    bins: (int)
        into how many bins are probabilities divided (default = 15)
    ece_full: (bool)
        whether to use ECE-full or ECE-max.

    Returns
    =======
    ece : float
        expected calibration error
    """

    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    # Prepare predictions, confidences and true labels for ECE calculation
    if ece_full:
        preds, confs, y_true = _get_preds_all(y_true, probs,
                                              normalize=normalize,
                                              flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence

    # Calculate ECE and ECE2
    ece = _ECE_helper(confs, preds, y_true, bin_size=1/bins, ece_full=ece_full)

    return ece


def _get_preds_all(y_true, y_probs, axis=1, normalize=False, flatten=True):
    """
    Method to get predictions in right format for ECE-full.

    Parameters
    ==========
    y_true: list
        containing the actual class labels
    y_probs: list (samples, classes)
        containing probabilities for all the classes
    axis: (int)
        dimension of set to calculate probabilities on
    normalize: (bool)
        in case of 1-vs-K calibration, the probabilities need to be
        normalized. (default = False)
    flatten: (bool)
        flatten all the arrays

    Returns
    =======
    (y_preds, y_probs, y_true)
        predictions, probabilities and true labels
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    elif len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    y_preds = np.argmax(y_probs, axis=axis)  # Maximum confidence as prediction
    y_preds = y_preds.reshape(-1, 1)

    if normalize:
        y_probs /= np.sum(y_probs, axis=axis)

    n_classes = y_probs.shape[1]
    y_preds = label_binarize(y_preds, classes=range(n_classes))
    y_true = label_binarize(y_true, classes=range(n_classes))

    if flatten:
        y_preds = y_preds.flatten()
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()

    return y_preds, y_probs, y_true


def _ECE_helper(conf, pred, true, bin_size=0.1, ece_full=False):

    """
    Expected Calibration Error

    Parameters
    ==========
    conf (numpy.ndarray):
        list of confidences
    pred (numpy.ndarray):
        list of predictions
    true (numpy.ndarray):
        list of true labels
    bin_size: (float):
        size of one bin (0,1)  # TODO should convert to number of bins?

    Returns
    =======
    ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Find accur. and confidences per bin
        acc, avg_conf, len_bin = _compute_acc_bin(conf_thresh-bin_size,
                                                  conf_thresh, conf, pred,
                                                  true, ece_full)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece


def _compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true,
                     ece_full=True):
    """
    # Computes accuracy and average confidence for bin

    Parameters
    ==========
    conf_thresh_lower (float):
        Lower Threshold of confidence interval
    conf_thresh_upper (float):
        Upper Threshold of confidence interval
    conf (numpy.ndarray):
        list of confidences
    pred (numpy.ndarray):
        list of predictions
    true (numpy.ndarray):
        list of true labels
    pred_thresh (float) :
        float in range (0,1), indicating the prediction threshold

    Returns
    =======
    (accuracy, avg_conf, len_bin) :
        accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf)
                       if (x[2] > conf_thresh_lower or conf_thresh_lower == 0)
                       and (x[2] <= conf_thresh_upper)]

    if len(filtered_tuples) < 1:
        return 0.0, 0.0, 0
    else:
        if ece_full:
            # How many elements falls into given bin
            len_bin = len(filtered_tuples)
            # Avg confidence of BIN
            avg_conf = sum([x[2] for x in filtered_tuples])/len_bin
            # Mean difference from actual class
            accuracy = np.mean([x[1] for x in filtered_tuples])
        else:
            # How many correct labels
            correct = len([x for x in filtered_tuples if x[0] == x[1]])
            # How many elements falls into given bin
            len_bin = len(filtered_tuples)
            # Avg confidence of BIN
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin
            # accuracy of BIN
            accuracy = float(correct)/len_bin

    return accuracy, avg_conf, len_bin


def _MCE_helper(conf, pred, true, bin_size=0.1, mce_full=True):

    """
    Maximal Calibration Error

    Parameters
    ==========
    conf (numpy.ndarray): list of confidences
    pred (numpy.ndarray): list of predictions
    true (numpy.ndarray): list of true labels
    bin_size: (float):
        size of one bin (0,1)  # TODO should convert to number of bins?
    mce_full: (bool)
        whether to use ECE-full or ECE-max for bin calculation

    Returns
    =======
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, count = _compute_acc_bin(conf_thresh-bin_size,
                                                conf_thresh, conf, pred, true,
                                                mce_full)
        cal_errors.append(np.abs(acc-avg_conf))

    return np.max(np.asarray(cal_errors))


def MCE(y_true, probs, normalize=False, bins=15, mce_full=False):

    """
    Calculate MCE score based on model output probabilities and true labels

    Parameters
    ==========
    y_true : list
        containing the actual class labels
    probs : list
        containing probabilities for all the classes with a shape of (samples,
        classes)
    normalize : bool
        in case of 1-vs-K calibration, the probabilities need to be normalized.
        (default = False)
    bins : int
        into how many bins are probabilities divided (default = 15)
    mce_full : boolean
        whether to use ECE-full or ECE-max for calculation MCE.

    Returns
    =======
    mce : float
        maximum calibration error
    """

    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(probs.shape) == len(y_true.shape):
        y_true = np.argmax(y_true, axis=1)

    # Prepare predictions, confidences and true labels for MCE calculation
    if mce_full:
        preds, confs, y_true = _get_preds_all(y_true, probs,
                                              normalize=normalize,
                                              flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence

    # Calculate MCE
    mce = _MCE_helper(confs, preds, y_true, bin_size=1/bins, mce_full=mce_full)

    return mce


def conf_MCE(y_true, probs, bins=15):
    """
    Calculate ECE score based on model output probabilities and true labels

    Parameters
    ==========
    y_true:
        - a list containing the actual class labels
        - ndarray shape (n_samples) with a list containing actual class
          labels
        - ndarray shape (n_samples, n_classes) with largest value in
          each row for the correct column class.
    probs:
        a list containing probabilities for all the classes with a shape of
        (samples, classes)
    bins: (int)
        - into how many bins are probabilities divided (default = 15)

    Returns
    =======
    mce : float
        maximum calibration error
    """
    return MCE(y_true, probs, normalize=False, bins=bins, mce_full=False)


def binary_ECE(y_true, probs, power=1, bins=15):
    r"""Binary Expected Calibration Error

    .. math::

        \text{binary-ECE}  = \sum_{i=1}^M \frac{|B_{i}|}{N} |
        \bar{y}(B_{i}) - \bar{p}(B_{i})|

    Parameters
    ----------
    y_true : indicator vector (n_samples, )
        True labels.

    probs : matrix (n_samples, )
        Predicted probabilities for positive class.

    Returns
    -------
    score : float

    Examples
    --------
    >>> from pycalib.metrics import binary_ECE
    >>> Y = np.array([0, 1])
    >>> P = np.array([0.1, 0.9])
    >>> print(round(binary_ECE(Y, P, bins=2), 8))
    0.1
    >>> Y = np.array([0, 0, 0, 1, 1, 1])
    >>> P = np.array([.1, .2, .3, .7, .8, .9])
    >>> print(round(binary_ECE(Y, P, bins=2), 8))
    0.2
    """
    idx = np.digitize(probs, np.linspace(0, 1 + 1e-8, bins + 1)) - 1

    def bin_func(y, p, idx):
        return ((np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power)
                * np.sum(idx) / len(p))

    ece = 0
    for i in np.unique(idx):
        # print('Mean scores', np.mean(probs[idx == i]))
        # print('True proportion', np.mean(y_true[idx == i]))
        # print('Difference ', np.abs(np.mean(probs[idx == i])
        #                      - np.mean(y_true[idx == i])))
        ece += bin_func(y_true, probs, idx == i)
    return ece


def classwise_ECE(y_true, probs, power=1, bins=15):
    r"""Classwise Expected Calibration Error

    .. math::

        \text{class-$j$-ECE}  = \sum_{i=1}^M \frac{|B_{i,j}|}{N}
        |\bar{y}_j(B_{i,j}) - \bar{p}_j(B_{i,j})|,

        \text{classwise-ECE}  = \frac{1}{K}\sum_{j=1}^K \text{class-$j$-ECE}

    Parameters
    ----------
    y_true : label indicator matrix (n_samples, n_classes)
        True labels.
        # TODO Add option to pass array with shape (n_samples, )

    probs : matrix (n_samples, n_classes)
        Predicted probabilities.

    Returns
    -------
    score : float

    Examples
    --------
    >>> from pycalib.metrics import classwise_ECE
    >>> Y = np.array([[1, 0], [0, 1]])
    >>> P = np.array([[0.9, 0.1], [0.1, 0.9]])
    >>> print(round(classwise_ECE(Y, P, bins=2), 8))
    0.1
    >>> Y = np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    >>> P = np.array([[.9, .8, .7, .3, .2, .1], [.1, .2, .3, .7, .8, .9]])
    >>> print(round(classwise_ECE(Y, P, bins=2), 8))
    0.2
    """
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true),
                                classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.mean(
        [
            binary_ECE(
                y_true[:, c].astype(float), probs[:, c], power=power, bins=bins
            ) for c in range(n_classes)
        ]
    )


def simplex_binning(y_true, probs, bins=15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true),
                                classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins + 1)) - 1

    prob_bins = {}
    label_bins = {}

    for i, row in enumerate(idx):
        try:
            prob_bins[','.join([str(r) for r in row])].append(probs[i])
            label_bins[','.join([str(r) for r in row])].append(y_true[i])
        except KeyError:
            prob_bins[','.join([str(r) for r in row])] = [probs[i]]
            label_bins[','.join([str(r) for r in row])] = [y_true[i]]

    bins = []
    for key in prob_bins:
        bins.append(
            [
                len(prob_bins[key]),
                np.mean(np.array(prob_bins[key]), axis=0),
                np.mean(np.array(label_bins[key]), axis=0)
            ]
        )

    return bins


def full_ECE(y_true, probs, bins=15, power=1):
    n = len(probs)

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true),
                                classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins + 1)) - 1

    filled_bins = np.unique(idx, axis=0)

    s = 0
    for bin in filled_bins:
        i = np.where((idx == bin).all(axis=1))[0]
        s += (len(i)/n) * (
            np.abs(np.mean(probs[i], axis=0) - np.mean(y_true[i],
                                                       axis=0))**power
        ).sum()

    return s


def _label_resampling(probs):
    c = probs.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    y = np.zeros_like(probs)
    y[range(len(probs)), choices] = 1
    return y


def _score_sampling(probs, samples=10000, ece_function=None):

    probs = np.array(probs)

    return np.array(
        [
            ece_function(probs, _label_resampling(probs)) for sample in
            range(samples)
        ]
    )


def pECE(y_true, probs, samples=10000, ece_function=full_ECE):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true),
                                classes=range(probs.shape[1]))

    return 1 - (
        percentileofscore(
            _score_sampling(
                probs,
                samples=samples,
                ece_function=ece_function
            ),
            ece_function(y_true, probs)
        ) / 100
    )
