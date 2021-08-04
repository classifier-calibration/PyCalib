import unittest
import numpy as np
from pycalib.models import (IsotonicCalibration, LogisticCalibration,
                            BinningCalibration, SigmoidCalibration)
from numpy.testing import assert_array_equal


class TestIsotonicCalibration(unittest.TestCase):
    def test_dummy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = IsotonicCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestLogisticCalibration(unittest.TestCase):
    def test_dummy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = LogisticCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestBinningCalibration(unittest.TestCase):
    def test_dummy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = BinningCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


class TestSigmoidCalibration(unittest.TestCase):
    def test_dummy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([1, 0])
        cal = SigmoidCalibration()
        cal.fit(S, Y)
        pred = cal.predict(S)
        assert_array_equal(Y, pred)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
