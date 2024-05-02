.. title:: Quickstart

.. _quickstart:

Quickstart
==========

The simplest way to calibrate an existing probabilistic classifier is the
following:

First choose the calibration method you want to use

.. code-block:: python

    from pycalib.models import IsotonicCalibration
    cal = IsotonicCalibration()

Now we can put together a probabilistic classifier with the chosen calibration
method

.. code-block:: python

    from pycalib.models import CalibratedModel

    cal_clf = CalibratedModel(base_estimator=clf, calibrator=cal)

Now you can train both classifier and calibrator all together.

.. code-block:: python

    from sklearn.datasets import load_iris

    dataset = load_iris()
    cal_clf.fit(dataset.data, dataset.target)

For a full example check the Section Examples quick start.
