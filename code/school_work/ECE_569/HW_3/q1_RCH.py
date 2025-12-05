# code for Question 1.b: Reduced C-Hull classifier on overlapping data
# paths are hardcoded for my ease, but of course they will have to be changed for someone else

import cvxpy as cp
import numpy as np
from scipy.io import loadmat

# load data: train_overlap
data = loadmat('code/school_work/ECE_569/HW_3/train_overlap.mat')

# create variables - Reduced C-Hull
# read in A, B from data files
# create u with dimension equal to number of columns in A
# create v with dimension equal to number of columns in B
# hyperparameter 0 < d <= 1
A = data['A']
B = data['B']
u = cp.Variable(A.shape[1], nonneg=True)
v = cp.Variable(B.shape[1], nonneg=True)
d = 0.75 # somewhat arbitrary choice

# create constraints
# 1^Tu = 1
# 1^Tv = 1
constraints = [
    cp.sum(u) == 1,
    cp.sum(v) == 1,
    u <= d,
    v <= d
]


# form objective
# minimize ||Au - Bv||_2^2
obj = cp.Minimize(cp.sum_squares(A @ u - B @ v))

# form and solve problem
prob = cp.Problem(obj, constraints)
prob.solve()
print('Reduced C-Hull status:', prob.status)

# evaluate on test data
data = loadmat('code/school_work/ECE_569/HW_3/test_overlap.mat')
X_test = data['X_test'].T
true_labels = data['true_labels'].flatten()

# form convex hulls based on optimized u, v
A_hull = A @ u.value
B_hull = B @ v.value

# find distances from test points to convex hulls
dist_to_A_hull = np.linalg.norm(X_test - A_hull, axis=1)
dist_to_B_hull = np.linalg.norm(X_test - B_hull, axis=1)

# class A = 1, class B = -1
pred_labels = np.where(dist_to_A_hull <= dist_to_B_hull, 1, -1)
print('Reduced C-Hull Test Accuracy:', np.mean(pred_labels == true_labels))






# data visualization
import matplotlib.pyplot as plt

# train_overlap plot
plt.figure(figsize=(8,8))

# plot training points
plt.scatter(A[0, :], A[1, :], color='blue', label='Class A (Train)', alpha=0.5)
plt.scatter(B[0, :], B[1, :], color='red', label='Class B (Train)', alpha=0.5)

# plot decision boundary
midpoint = (A_hull + B_hull)/2
dx, dy = B_hull - A_hull
slope = -dx/dy if dy != 0 else 1e6  # vertical line
intercept = midpoint[1] - slope*midpoint[0]

# limit plotting space (don't zoom out too far)
x_min = (A[0,:].min() + B[0,:].min()) / 2
x_max = (A[0,:].max() + B[0,:].max()) / 2
y_min = (A[1,:].min() + B[1,:].min()) / 2
y_max = (A[1,:].max() + B[1,:].max()) / 2

x_vals = np.linspace(x_min, x_max, 100)
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, 'g--', linewidth=3, label='Decision boundary')

plt.title('Reduced C-Hull Classifier: Overlapping Data (Train)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')
plt.tight_layout()
# plt.show()










# Compute perpendicular bisector for decision boundary
midpoint = (A_hull + B_hull) / 2
dx, dy = B_hull - A_hull

slope = -dx / dy
intercept = midpoint[1] - slope * midpoint[0]
x_vals = np.linspace(X_test[:, 0].min() / 2, X_test[:, 0].max() / 2, 100)
y_vals = slope * x_vals + intercept

plt.figure(figsize=(8,8))

# Plot test points by true label
plt.scatter(X_test[true_labels==1,0], X_test[true_labels==1,1], color='blue', label='Class A', alpha=0.5)
plt.scatter(X_test[true_labels==-1,0], X_test[true_labels==-1,1], color='red', label='Class B', alpha=0.5)

# Plot decision boundary
plt.plot(x_vals, y_vals, 'green', linestyle='--', linewidth=3, label='Decision boundary')

plt.axis('equal')
plt.tight_layout()

plt.title('Reduced C-Hull Classifier: Overlapping Data (Test)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()