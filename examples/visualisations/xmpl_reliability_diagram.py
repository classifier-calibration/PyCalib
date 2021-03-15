"""
=============================
Plotting reliability diagrams
=============================

This example illustrates how to visualise the reliability diagram for a binary
probabilistic classifier.
"""
# Author: Miquel Perello Nieto <miquel.perellonieto@bristol.ac.uk>
# License: new BSD

from pprint import pprint

print(__doc__)

##############################################################################
# This example shows different ways to visualise the reliability diagram for a
# binary classification problem.
# 
# First we will generate two synthetic models and some synthetic scores and
# labels.

import matplotlib.pyplot as plt
import numpy as np

n_c1 = n_c2 = 200
p = np.concatenate((np.random.beta(2, 5, n_c1),
                    np.random.beta(4, 3, n_c2)
                   ))

y = np.concatenate((np.zeros(n_c1), np.ones(n_c2)))

print(p.shape)
print(y.shape)

s1 = 1/(1 + np.exp(-3*(p - 0.5)))
s2 = 1/(1 + np.exp(-8*(p - 0.5)))

plt.scatter(s1, p, label='Model 1')
plt.scatter(s2, p, label='Model 2')
plt.scatter(p, y)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('Model scores')
plt.ylabel('Sample true probability')
plt.grid()
plt.legend()

p = np.vstack((1 - p, p)).T
s1 = np.vstack((1 - s1, s1)).T
s2 = np.vstack((1 - s2, s2)).T

##############################################################################
# A perfect calibration should be as follows, compared with the generated
# scores

import scipy.stats as stats

p_g_p = stats.beta.pdf(x=p[:, 1], a=3, b=2)
p_g_n = stats.beta.pdf(x=p[:, 1], a=2, b=7)

p_hat = p_g_p/(p_g_n+p_g_p)
p_hat = np.vstack((1 - p_hat, p_hat)).T

plt.scatter(p[:, 1], s1[:, 1], label='Model 1')
plt.scatter(p[:, 1], s2[:, 1], label='Model 2')
plt.scatter(p[:, 1], p_hat[:, 1], color='red', label='Bayes optimal correction')
plt.xlabel('Sample true probability')
plt.ylabel('Model scores')
plt.grid()
plt.legend()

##############################################################################
# Then we can show the most common form to visualise a reliability diagram

from pycalib.visualisations import plot_reliability_diagram

fig = plt.figure(figsize=(7, 7))
fig = plot_reliability_diagram(labels=y, scores_list=[s1, ],
                               legend=['Model 1'], bins=10,
                               class_names=['Negative', 'Positive'], fig=fig)

##############################################################################
# We can enable some parameters to show several aspects of the reliability
# diagram. For example, we can add a histogram indicating the number of samples
# on each bin (or show the count in each marker), the correction that should be
# applied to the average scores in order to calibrate the model can be also
# shown as red arrows pointing to the direction of the diagonal (perfectly
# calibrated model). And even the true class of each sample at the y
# coordinates [0 and 1] for each scored instance.

from pycalib.visualisations import plot_reliability_diagram

fig = plt.figure(figsize=(7, 7))
fig = plot_reliability_diagram(labels=y, scores_list=[s1, ],
                               legend=['Model 1'],
                               show_histogram=True,
                               bins=10, class_names=['Negative', 'Positive'],
                               fig=fig, show_counts=True,
                               show_correction=True,
                               show_samples=True,
                               sample_proportion=1.0,
                               hist_per_class=True)

##############################################################################
# It can be also useful to have 95% confidence intervals for each bin by
# performing a binomial proportion confidence interval with various statistical
# tests. This function uses https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
# thus accepts the different tests available in the statsmodels library. In the
# following example we use the Clopper-Pearson interval based on Beta
# distribution and a confidence interval of 95%.

fig = plt.figure(figsize=(7, 7))
fig = plot_reliability_diagram(labels=y, scores_list=[s1, ],
                               legend=['Model 1'],
                               show_histogram=True,
                               bins=10, class_names=['Negative', 'Positive'],
                               fig=fig,
                               show_samples=True, sample_proportion=1.0,
                               errorbar_interval=0.95,
                               interval_method='beta',)

##############################################################################
# The function also allows the visualisation of multiple models for comparison.

fig = plt.figure(figsize=(7, 7))
fig = plot_reliability_diagram(labels=y, scores_list=[s1, s2],
                               legend=['Model 1', 'Model 2'],
                               show_histogram=True,
                               bins=10, class_names=['Negative', 'Positive'],
                               fig=fig,
                               errorbar_interval=0.95,
                               interval_method='beta')
