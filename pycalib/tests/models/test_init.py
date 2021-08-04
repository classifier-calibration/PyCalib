import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from pycalib.models import (IsotonicCalibration, LogisticCalibration,
                            BinningCalibration, SigmoidCalibration,
                            CalibratedModel)
from numpy.testing import assert_array_equal


class TestIsotonicCalibration(unittest.TestCase):
    def test_fit_predict(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = IsotonicCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestLogisticCalibration(unittest.TestCase):
    def test_fit_predict(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = LogisticCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestBinningCalibration(unittest.TestCase):
    def test_fit_predict(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = BinningCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestSigmoidCalibration(unittest.TestCase):
    def test_fit_predict(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = SigmoidCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestCalibratedModel(unittest.TestCase):
    def test_fit_predict(self):
        X, Y = make_blobs(n_samples=10000, centers=5, n_features=2,
                                    random_state=42)
        Y = (Y > 2).astype(int)
        cal = CalibratedModel(LogisticRegression(), IsotonicCalibration())
        cal.fit(X, Y)

        pred = cal.predict(X)
        self.assertGreater(np.mean(Y == pred), 0.7)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
