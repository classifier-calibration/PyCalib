.. title:: Quickstart

.. _quickstart:

Quickstart
==========

The simplest way to calibrate an existing probabilistic classifier is the
following:

First choose the calibration method you want to use

```
from pycalib.models import IsotonicCalibration
cal = IsotonicCalibration()
```

Now we can put together a probabilistic classifier with the chosen calibration
method

```
from pycalib.models import CalibratedModel

cal_clf = CalibratedModel(base_estimator=clf, method=cal)
```

Now you can train both classifier and calibrator all together.

```
from sklearn.datasets import load_iris

dataset = load_iris()
cal_clf.fit(dataset.data, dataset.target)
```
