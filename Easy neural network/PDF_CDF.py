# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 11:54:28 2021

@author: palusbar
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 7, 100)
pdf = stats.norm.pdf(x, loc = 1, scale = 2**(1/2))
fig, axes = plt.subplots()
axes.plot(x, pdf, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('PDF')


cdf = stats.norm.cdf(x, loc = 1, scale = 2**(1/2))
fig, axes = plt.subplots()
axes.plot(x, cdf, 'r')
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('CDF')

boundary2 = stats.norm.cdf(2, loc = 1, scale = 2**(1/2)) - stats.norm.cdf(0.5, loc = 1, scale = 2**(1/2))

print(boundary2)

