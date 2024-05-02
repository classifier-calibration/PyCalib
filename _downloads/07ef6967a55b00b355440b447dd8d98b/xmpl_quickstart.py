"""
=============================
Quickstart
=============================

This example shows a simple comparison of the expected calibration error of a
non-calibrated method against a calibrated method.
"""
# Author: Miquel Perello Nieto <miquel.perellonieto@bristol.ac.uk>
# License: new BSD

print(__doc__)

##############################################################################
# First choose a classifier

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

##############################################################################
# And a dataset

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=100000, n_features=20, n_informative=4, n_redundant=4,
    random_state=42
)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y)

##############################################################################
# We can see how calibrated it is after training

clf.fit(X_train, Y_train)

n_correct = sum(clf.predict(X_test) == Y_test)
n_test = Y_test.shape[0]

print(f"The classifier gets {n_correct} correct "
      f"predictions out of {n_test}")

##############################################################################
# We can asses the confidence expected calibration error

from pycalib.metrics import conf_ECE

scores = clf.predict_proba(X_test)
cece = conf_ECE(Y_test, scores, bins=15)

print(f"The classifier gets a confidence expected "
      f"calibration error of {cece:0.2f}")

##############################################################################
# Let's look at its reliability diagram

from pycalib.visualisations import plot_reliability_diagram

plot_reliability_diagram(labels=Y_test, scores=scores, show_histogram=True,
                         show_bars=True, show_gaps=True)

##############################################################################
# We can see how a calibration can improve the conf-ECE

from pycalib.models import IsotonicCalibration
cal = IsotonicCalibration()

##############################################################################
# Now we can put together a probabilistic classifier with the chosen calibration
# method

from pycalib.models import CalibratedModel

cal_clf = CalibratedModel(base_estimator=clf, calibrator=cal,
                          fit_estimator=False)

##############################################################################
# Now you can train both classifier and calibrator all together.

cal_clf.fit(X_train, Y_train)
n_correct = sum(cal_clf.predict(X_test) == Y_test)

print(f"The calibrated classifier gets {n_correct} "
      f"correct predictions out of {n_test}")

scores_cal = cal_clf.predict_proba(X_test)
cece = conf_ECE(Y_test, scores_cal, bins=15)

print(f"The calibrated classifier gets a confidence "
      f"expected calibration error of {cece:0.2f}")

##############################################################################
# Now you can train both classifier and calibrator all together.

from pycalib.visualisations import plot_reliability_diagram

plot_reliability_diagram(labels=Y_test, scores=scores_cal, show_histogram=True,
                         show_bars=True, show_gaps=True)
