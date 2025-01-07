import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('TKAgg')

# Load data
train_data = pd.read_csv('C:/Users/Admin/Documents/0TL/ml/BTL/data/train 100.csv')
test_data = pd.read_csv('C:/Users/Admin/Documents/0TL/ml/BTL/data/test 100.csv')

# Let's use two most important features for visualization: battery_power and ram
X_train = train_data[['battery_power', 'ram']].values
y_train = train_data['price_range'].values
X_test = test_data[['battery_power', 'ram']].values

# Create mesh grid for visualization
x_min, x_max = X_train[:, 0].min() - 100, X_train[:, 0].max() + 100
y_min, y_max = X_train[:, 1].min() - 100, X_train[:, 1].max() + 100
xx, yy = np.meshgrid(np.arange(x_min, x_max, 100),
                     np.arange(y_min, y_max, 100))

# Load trained SVM weights and bias
w = np.load('svm_weights.npy')
b = np.load('svm_bias.npy')

# Create decision boundary
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w[:2]) + b
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z > 0, alpha=0.4)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.8)
plt.xlabel('Battery Power')
plt.ylabel('RAM')
plt.title('SVM Decision Boundary (Battery Power vs RAM)')
plt.colorbar(label='Price Range')
plt.show()