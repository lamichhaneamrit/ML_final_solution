import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

cwd = os.getcwd()
#sys.path.append(r'C:\Users\tusha\OneDrive\Desktop\Machine Learning I')
# #################################################################################################################
# Example 3: TASK A

N = 100
mu = 20
standard_dev = 4
sigma = math.sqrt(standard_dev)
s = np.random.normal(mu, sigma, N)
gauss_func = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (s - mu)**2 / (2 * sigma**2))

plt.figure()
plt.subplot(2,1,1)
plt.scatter(s, gauss_func, alpha=0.5)
plt.xlabel('random values')
plt.ylabel('gaussian pdf')
plt.grid()


plt.subplot(2,1,2)
count, bins, ignored = plt.hist(s, 20, density=True, alpha=0.6, color='g')
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
plt.xlabel('random values')
plt.ylabel('gaussian pdf')
plt.grid()
plt.show()

# #################################################################################################################
# Example 3: TASK B

MLE_mu = (1/N)*(np.sum(s))
MLE_sigma = (1/N)*(np.sum((s - MLE_mu)**2))

print('Maximum likelihood of the mean: ', MLE_mu)
print('Maximum likelihood of the variance :', MLE_sigma)

# #################################################################################################################
# Example 3: TASK D

# For small number of data points, the distribution will be same as of with larger number of data points, but the
# variation among the sample means will be larger.

# ##################################################################################################################






