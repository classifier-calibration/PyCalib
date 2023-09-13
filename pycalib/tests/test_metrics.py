import unittest
import numpy as np
from functools import partial
from pycalib.metrics import (accuracy, cross_entropy, brier_score,
                             binary_ECE, conf_ECE, classwise_ECE, full_ECE,
                             MCE, pECE)

from sklearn.preprocessing import label_binarize


# TODO add more test cases
class TestFunctions(unittest.TestCase):
    def test_accuracy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[0, 1], [0, 1]])
        acc = accuracy(Y, S)
        self.assertAlmostEqual(acc, 0.5)

        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[1, 0], [0, 1]])
        acc = accuracy(Y, S)
        self.assertAlmostEqual(acc, 0.0)

    def test_cross_entropy(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[0, 1], [0, 1]])
        ce = cross_entropy(Y, S)
        expected = - (np.log(0.9) + np.log(0.4))/2
        self.assertAlmostEqual(ce, expected)

        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[1, 0], [0, 1]])
        ce = cross_entropy(Y, S)
        expected = - (np.log(0.1) + np.log(0.4))/2
        self.assertAlmostEqual(ce, expected)

    def test_brier_score(self):
        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[0, 1], [0, 1]])
        bs = brier_score(Y, S)
        expected = np.mean(np.abs(S - Y)**2)
        self.assertAlmostEqual(bs, expected)

        S = np.array([[0.1, 0.9], [0.6, 0.4]])
        Y = np.array([[1, 0], [0, 1]])
        bs = brier_score(Y, S)
        expected = np.mean(np.abs(S - Y)**2)
        self.assertAlmostEqual(bs, expected)

    def test_binary_ece(self):
        S = np.array([.6, .6, .6, .6, .6, .6, .6, .6, .6, .6])
        y = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        ece = binary_ECE(y, S)
        self.assertAlmostEqual(ece, 0)

    def test_conf_ece(self):
        S = np.array([[0.6, 0.2, 0.2]]*10)
        y = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2]
        Y = label_binarize(y, classes=range(3))
        cece = conf_ECE(Y, S)
        self.assertAlmostEqual(cece, 0)
        # TODO Add more tests

    def test_classwise_ece(self):
        S = np.array([[0.6, 0.2, 0.2]]*10)
        Y = label_binarize([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], classes=range(3))
        ece = classwise_ECE(Y, S)
        self.assertAlmostEqual(ece, 0)
        # TODO Add more tests

    def test_full_ece(self):
        S = np.array([[0.6, 0.2, 0.2]]*10)
        Y = label_binarize([0, 0, 0, 0, 0, 0, 1, 1, 2, 2], classes=range(3))
        ece = full_ECE(Y, S)
        self.assertAlmostEqual(ece, 0)
        # TODO Add more tests

    def test_conf_mce(self):
        S = np.ones((2, 3))/3.0
        y = np.array([0, 0])
        mce = MCE(y, S)
        self.assertAlmostEqual(mce, 2.0/3)

        y = np.array([0, 1, 2])
        S = np.array([[1/3, 0.3, 0.3],
                      [1/3, 0.3, 0.3],
                      [1/3, 0.3, 0.3]])
        mce = MCE(y, S)
        self.assertAlmostEqual(mce, 0.0)

        y = np.array([0, 1, 2])
        S = np.array([[0.3, 1/3, 0.3],
                      [0.3, 1/3, 0.3],
                      [0.3, 1/3, 0.3]])
        mce = MCE(y, S)
        self.assertAlmostEqual(mce, 0.0)

        y = np.array([0, 1, 2])
        S = np.array([[0.3, 0.3, 1/3],
                      [0.3, 0.3, 1/3],
                      [0.3, 0.3, 1/3]])
        mce = MCE(y, S)
        self.assertAlmostEqual(mce, 0.0)

        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        S = np.array([[0.3, 0.3, 1/3],
                      [0.3, 0.3, 1/3],
                      [0.3, 0.3, 1/3]])
        mce = MCE(Y, S)
        self.assertAlmostEqual(mce, 0.0)

        Y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
        S = np.array([[0.4, 0.3, 0.3],  # correct
                      [0.3, 0.4, 0.3],  # incorrect
                      [0.3, 0.3, 0.4],  # incorrect
                      [0.3, 0.3, 0.4],  # incorrect

                      [0.1, 0.7, 0.2],  # incorrect mean conf 0.75
                      [0.2, 0.1, 0.7],  # incorrect
                      [0.2, 0.8, 0.2],  # incorrect
                      [0.8, 0.1, 0.1]   # incorrect
                      ])
        mce = MCE(Y, S, bins=2)
        self.assertEqual(mce, 0.75)

        Y = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0],
                      [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
        S = np.array([[0.4, 0.3, 0.3],  # correct   # conf 0.4
                      [0.3, 0.4, 0.3],  # incorrect
                      [0.3, 0.3, 0.4],  # incorrect
                      [0.3, 0.3, 0.4],  # incorrect

                      [0.1, 0.7, 0.2],  # incorrect
                      [0.7, 0.1, 0.2],  # correct
                      [0.8, 0.0, 0.2],  # correct
                      [0.1, 0.8, 0.1]   # correct
                      ])
        mce = MCE(Y, S, bins=2)
        self.assertAlmostEqual(mce, 0.4 - 1/4)


    def test_calibrated_p_ece(self):
        p = np.random.rand(5000, 3)
        p /= p.sum(axis=1)[:, None]
        multinomial = partial(np.random.multinomial, 1)
        y = np.apply_along_axis(multinomial, 1, p)
        calibrated_pECE = pECE(y, p, samples=2000, ece_function=classwise_ECE)
        self.assertGreater(calibrated_pECE, 0.04)
        calibrated_pECE = pECE(y, p, samples=2000, ece_function=conf_ECE)
        self.assertGreater(calibrated_pECE, 0.04)

    def test_uncalibrated_p_ece(self):
        p = np.random.rand(1000, 3)
        p /= p.sum(axis=1)[:, None]
        y = np.eye(3)[np.random.choice([0, 1, 2], size=p.shape[0])]
        uncalibrated_pECE = pECE(y, p, samples=1000, ece_function=classwise_ECE)
        self.assertLess(uncalibrated_pECE, 0.04)
        uncalibrated_pECE = pECE(y, p, samples=1000, ece_function=conf_ECE)
        self.assertLess(uncalibrated_pECE, 0.04)



def main():
    unittest.main()


if __name__ == '__main__':
    main()
