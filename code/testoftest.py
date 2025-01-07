# Import necessary libraries
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

matplotlib.use('TkAgg')

# Generate sample data
np.random.seed(0)
mean1 = [0, 0]
cov1 = [[0.5, 0.1], [0.1, 0.5]]
mean2 = [5, 5]
cov2 = [[1, 0.5], [0.5, 1]]

data1 = np.random.multivariate_normal(mean1, cov1, 100)
data2 = np.random.multivariate_normal(mean2, cov2, 100)

X = np.vstack((data1, data2))
Y = np.hstack((np.zeros(100), np.ones(100)))

# Train SVM model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# Plot data and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], c='blue', label='Class 1')
plt.scatter(data2[:, 0], data2[:, 1], c='red', label='Class 2')

# Plot decision boundary (hyperplane)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(X[:, 0].min(), X[:, 0].max())
yy = a * xx - clf.intercept_[0] / w[1]
plt.plot(xx, yy, 'k-')

# Plot margin
yy_down = yy + 1 / w[1]
yy_up = yy - 1 / w[1]
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.legend()
plt.show()