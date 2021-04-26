import unittest
import matplotlib.pyplot as plt
import numpy as np

from pycalib.visualisations import plot_reliability_diagram


class TestVisualisations(unittest.TestCase):
    def test_plot_reliability_diagram(self):
        n_c1 = n_c2 = 500
        p = np.concatenate((np.random.beta(2, 5, n_c1),
                            np.random.beta(4, 3, n_c2)))

        y = np.concatenate((np.zeros(n_c1), np.ones(n_c2)))

        s1 = 1/(1 + np.exp(-3*(p - 0.5)))
        s2 = 1/(1 + np.exp(-8*(p - 0.5)))

        p = np.vstack((1 - p, p)).T
        s1 = np.vstack((1 - s1, s1)).T
        s2 = np.vstack((1 - s2, s2)).T

        fig = plot_reliability_diagram(labels=y, scores=[s1, s2])
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_reliability_diagram_confidence(self):
        n_c1 = n_c2 = 500
        p = np.concatenate((np.random.beta(2, 5, n_c1),
                            np.random.beta(4, 3, n_c2)))

        y = np.concatenate((np.zeros(n_c1), np.ones(n_c2)))

        s1 = 1/(1 + np.exp(-3*(p - 0.5)))
        s2 = 1/(1 + np.exp(-8*(p - 0.5)))

        p = np.vstack((1 - p, p)).T
        s1 = np.vstack((1 - s1, s1)).T
        s2 = np.vstack((1 - s2, s2)).T

        fig = plot_reliability_diagram(labels=y, scores=[s1, s2],
                                       confidence=True)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_reliability_diagram_simple(self):
        n_c1 = n_c2 = 500
        p = np.concatenate((np.random.beta(2, 5, n_c1),
                            np.random.beta(4, 3, n_c2)))

        y = np.concatenate((np.zeros(n_c1), np.ones(n_c2)))

        s1 = 1/(1 + np.exp(-3*(p - 0.5)))
        s2 = 1/(1 + np.exp(-8*(p - 0.5)))

        p = np.vstack((1 - p, p)).T
        s1 = np.vstack((1 - s1, s1)).T
        s2 = np.vstack((1 - s2, s2)).T

        fig = plot_reliability_diagram(labels=y, scores=[s1, s2],
                                       show_histogram=False)
        self.assertIsInstance(fig, plt.Figure)

        fig = plot_reliability_diagram(labels=y, scores=s2,
                                       show_histogram=True)
        self.assertIsInstance(fig, plt.Figure)


    def test_plot_reliability_diagram_full(self):
        n_c1 = n_c2 = 500
        p = np.concatenate((np.random.beta(2, 5, n_c1),
                            np.random.beta(4, 3, n_c2)
                           ))

        y = np.concatenate((np.zeros(n_c1), np.ones(n_c2)))

        s1 = 1/(1 + np.exp(-3*(p - 0.5)))
        s2 = 1/(1 + np.exp(-8*(p - 0.5)))
        s1 = np.vstack((1 - s1, s1)).T
        s2 = np.vstack((1 - s2, s2)).T

        fig = plot_reliability_diagram(labels=y, scores=s1,
                               legend=['Model 1'],
                               show_histogram=True,
                               bins=9, class_names=['Negative', 'Positive'],
                               show_counts=True,
                               show_correction=True,
                               show_gaps=True,
                               sample_proportion=0.5,
                               errorbar_interval=0.95,
                               hist_per_class=True)
        self.assertIsInstance(fig, plt.Figure)

        class_2_idx = range(int(len(y)/3), int(2*len(y)/3))
        y[class_2_idx] = 2
        s1 = np.hstack((s1, s1[:, 1].reshape(-1, 1)))
        s1[class_2_idx,2] *= 3
        s1 /= s1.sum(axis=1)[:, None]
        s2 = np.hstack((s2, s2[:, 1].reshape(-1, 1)))
        s2[class_2_idx,2] *= 2
        s2 /= s2.sum(axis=1)[:, None]

        bins = [0, .3, .5, .8, 1]
        fig = plot_reliability_diagram(labels=y, scores=[s1, s2],
                                       legend=['Model 3', 'Model 4'],
                                       show_histogram=True,
                                       show_correction=True,
                                       show_counts=True,
                                       show_bars=True,
                                       sample_proportion=0.3,
                                       bins=bins,
                                       color_list=['darkgreen', 'chocolate'],
                                       invert_histogram=True)
        self.assertIsInstance(fig, plt.Figure)

        fig = plot_reliability_diagram(labels=y, scores=[s1, s2],
                                       legend=['Model 3', 'Model 4'],
                                       show_histogram=True,
                                       show_correction=True,
                                       show_counts=True,
                                       sample_proportion=0.3,
                                       bins=bins,
                                       color_list=['darkgreen', 'chocolate'],
                                       invert_histogram=True,
                                       confidence=True)
        self.assertIsInstance(fig, plt.Figure)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
