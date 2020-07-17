from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
import numpy as np

def cross_entropy(y, y_pred):
    '''
    y : true y
    y_pred : predicted y
    '''
    print('Warning: The order of parameters y y_pred has recently chagned')
    return log_loss(y, y_pred)


def brier_score(y, y_pred):
    '''
    y : true y
    y_pred : predicted y
    '''
    print('Warning: The order of parameters y y_pred has recently chagned')
    return mean_squared_error(y, y_pred)

# Markus functions
from sklearn.preprocessing import label_binarize

# FIXME Follow scikit-learn convention of y_true as first argument
def guo_ECE(probs, y_true, bins=15):
    """
    Calculate ECE score based on model output probabilities and true labels

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: - a list containing the actual class labels
                - ndarray shape (n_samples) with a list containing actual class
                labels
                - ndarray shape (n_samples, n_classes) with largest value in
                each row for the correct column class.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        ece - expected calibration error
    """
    return ECE(probs, y_true, normalize=False, bins=bins, ece_full=False)

def ECE(probs, y_true, normalize = False, bins = 15, ece_full = True):
    """
    Calculate ECE score based on model output probabilities and true labels

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: - a list containing the actual class labels
                - ndarray shape (n_samples) with a list containing actual class
                labels
                - ndarray shape (n_samples, n_classes) with largest value in
                each row for the correct column class.
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized. (default = False)
        bins: (int) - into how many bins are probabilities divided (default = 15)
        ece_full: (bool) - whether to use ECE-full or ECE-max.

    Returns:
        ece - expected calibration error
    """

    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    # Prepare predictions, confidences and true labels for ECE calculation
    if ece_full:
        preds, confs, y_true = get_preds_all(probs, y_true, normalize=normalize, flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence


    # Calculate ECE and ECE2
    ece = ECE_helper(confs, preds, y_true, bin_size = 1/bins, ece_full = ece_full)

    return ece



def get_preds_all(y_probs, y_true, axis = 1, normalize = False, flatten = True):
    """
    Method to get predictions in right format for ECE-full.

    Params:
        y_probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        axis: (int) dimension of set to calculate probabilities on
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized. (default = False)
        flatten: (bool) - flatten all the arrays

    Returns:
        (y_preds, y_probs, y_true) - predictions, probabilities and true labels
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    elif len(y_true.shape) == 2 and y_true.shape[1] > 1:
        y_true = y_true.argmax(axis=1).reshape(-1, 1)

    y_preds = np.argmax(y_probs, axis=axis)  # Take maximum confidence as prediction
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


def ECE_helper(conf, pred, true, bin_size = 0.1, ece_full = False):

    """
    Expected Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true, ece_full)
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE

    return ece


def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true,
                    ece_full=True):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        pred_thresh (float) : float in range (0,1), indicating the prediction threshold

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if  (x[2] > conf_thresh_lower or conf_thresh_lower == 0)  and x[2] <= conf_thresh_upper]

    if len(filtered_tuples) < 1:
        return 0.0, 0.0, 0
    else:
        if ece_full:
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples])/len_bin  # Avg confidence of BIN
            accuracy = np.mean([x[1] for x in filtered_tuples])  # Mean difference from actual class
        else:
            correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
            len_bin = len(filtered_tuples)  # How many elements falls into given bin
            avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
            accuracy = float(correct)/len_bin  # accuracy of BIN

    return accuracy, avg_conf, len_bin


def MCE_helper(conf, pred, true, bin_size = 0.1, mce_full = True):

    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        mce_full: (bool) - whether to use ECE-full or ECE-max for bin calculation

    Returns:
        mce: maximum calibration error
    """

    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, count = compute_acc_bin(conf_thresh-bin_size,
                                               conf_thresh, conf, pred, true,
                                               mce_full)
        cal_errors.append(np.abs(acc-avg_conf))

    return np.max(np.asarray(cal_errors))


def MCE(probs, y_true, normalize=False, bins=15, mce_full=False):

    """
    Calculate MCE score based on model output probabilities and true labels

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized. (default = False)
        bins: (int) - into how many bins are probabilities divided (default = 15)
        mce_full: (bool) - whether to use ECE-full or ECE-max for calculation MCE.

    Returns:
        mce: maximum calibration error
    """

    probs = np.array(probs)
    y_true = np.array(y_true)
    if len(probs.shape) == len(y_true.shape):
        y_true = np.argmax(y_true, axis=1)

    # Prepare predictions, confidences and true labels for MCE calculation
    if mce_full:
        preds, confs, y_true = get_preds_all(probs, y_true,
                                             normalize=normalize, flatten=True)

    else:
        preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction

        if normalize:
            confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            confs = np.max(probs, axis=1)  # Take only maximum confidence

    # Calculate MCE
    mce = MCE_helper(confs, preds, y_true, bin_size=1/bins, mce_full=mce_full)

    return mce


def binary_ECE(probs, y_true, power = 1, bins = 15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power) * np.sum(idx) / len(probs)

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece

def classwise_ECE(probs, y_true, power = 1, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.sum(
        [
            binary_ECE(
                probs[:, c], y_true[:, c].astype(float), power = power, bins = bins
            ) for c in range(n_classes)
        ]
    )


def simplex_binning(probs, y_true, bins = 15):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

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


def full_ECE(probs, y_true, bins = 15, power = 1):
    n = len(probs)

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    filled_bins = np.unique(idx, axis=0)

    s = 0
    for bin in filled_bins:
        i = np.where((idx == bin).all(axis=1))[0]
        s += (len(i)/n) * (
            np.abs(np.mean(probs[i], axis=0) - np.mean(y_true[i], axis=0))**power
        ).sum()

    return s


def label_resampling(probs):
    c = probs.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    y = np.zeros_like(probs)
    y[range(len(probs)), choices] = 1
    return y


def score_sampling(probs, samples = 10000, ece_function = None):

    probs = np.array(probs)

    return np.array(
        [
            ece_function(probs, label_resampling(probs)) for sample in range(samples)
        ]
    )


def pECE(probs, y_true, samples = 10000, ece_function = full_ECE):

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    return 1 - (
        percentileofscore(
            score_sampling(
                probs,
                samples=samples,
                ece_function=ece_function
            ),
            ece_function(probs, y_true)
        ) / 100
    )
