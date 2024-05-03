.. title:: Quickstart

.. _quickstart:

Quickstart
==========

This is a simple description of how to calibrate a classifier using this
library. For an extended example check the Section Examples
:ref:`sphx_glr_examples_xmpl_quickstart.py`.

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

