# ADMM for C-hull classifier

# =================================================
# pseudo code:
# =================================================
'''
given rho
initialize:
    u, x as uniform
    v, y as uniform
    lamda_u = lambda_v = 0
    Lu = (2 * A^T * A + rho * I)^-1
    Lv = (2 * B^T * B + rho * I)^-1
    
    for i < MAX_ITER:
        u = Lu * (2A^TBv + rho(x - lambda_u))
        v = Lv * (2B^TAu + rho(y - lambda_v))
        
        x_old = x
        y_old = y
        
        x = project_to_simplex(u + lambda_u)
        y = project_to_simplex(v + lambda_v)
        
        lambda_u += u - x
        lambda_v += v - y
        
        prim = norm(u - x) + norm(v - y)
        dual = rho * norm(x - x_old) + norm(y - y_old)
        
        (convergence check with prim and dual)
        (exceeding MAX_ITER check)
        
    when done, u,v will be optimal
'''

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
'''
Projection of vector v onto simplex (this function is not my code, but copied from somwhere online)
Based on the algorithm by Duchi et al. (2008).
'''
def project_to_simplex(v):
    v = np.asarray(v)
    n = v.size

    # Sort v in descending order
    # Compute cumulative sums
    # Find the rho value
    # Compute theta
    # Compute projection
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0].max()
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)

    return w


# read data
data = loadmat('code/school_work/ECE_569/HW_3/train_separable.mat')
A = data['A']
B = data['B']

# variables
u = np.ones(A.shape[1]) / A.shape[1]
x = np.ones(A.shape[1]) / A.shape[1]
v = np.ones(B.shape[1]) / B.shape[1]
y = np.ones(B.shape[1]) / B.shape[1]

# parameters + limiters
rho = 50.0
tau_inc = 2.0
tau_dec = 2.0
mu = 10.0
TOL = 1e-6
MAX_ITER = 50_000

# scaled duals
lambda_u = np.zeros(A.shape[1])
lambda_v = np.zeros(B.shape[1])

# lagrangian
Lu = np.linalg.inv(2 * A.T @ A + rho * np.eye(A.shape[1]))
Lv = np.linalg.inv(2 * B.T @ B + rho * np.eye(B.shape[1]))

for i in range(MAX_ITER):

    # u and v updates
    u = Lu @ (2 * A.T @ B @ v + rho * (x - lambda_u))
    v = Lv @ (2 * B.T @ A @ u + rho * (y - lambda_v))

    # store old x, y for dual residual
    x_old = x
    y_old = y

    # x,y = projection onto simplex of 
    x = project_to_simplex(u + lambda_u)
    y = project_to_simplex(v + lambda_v)

    # dual variable update
    lambda_u = lambda_u + (u - x)
    lambda_v = lambda_v + (v - y)

    # compute residuals 
    prim = np.sqrt(np.linalg.norm(u - x)**2 + np.linalg.norm(v - y)**2)
    dual = rho * np.sqrt(np.linalg.norm(x - x_old)**2 + np.linalg.norm(y - y_old)**2)
    
    # potentially vary rho
    if np.linalg.norm(v) > mu * np.linalg.norm(u):
        rho *= tau_inc
    elif np.linalg.norm(u) > mu * np.linalg.norm(v):
        rho /= tau_dec

    # convergence check
    if prim < TOL and dual < TOL:
        break
    elif i == MAX_ITER - 1:
        print("Reached maximum iterations without convergence.")


# after convergence:
# u and v are the optimal convex-hull weights

        
        
# evaluate on test data
data = loadmat('code/school_work/ECE_569/HW_3/test_separable.mat')
X_test = data['X_test'].T
true_labels = data['true_labels'].flatten()

# form convex hulls based on optimized u, v
A_hull = A @ u
B_hull = B @ v

# find distances from test points to convex hulls
dist_to_A_hull = np.linalg.norm(X_test - A_hull, axis=1)
dist_to_B_hull = np.linalg.norm(X_test - B_hull, axis=1)

# class A = 1, class B = -1
pred_labels = np.where(dist_to_A_hull <= dist_to_B_hull, 1, -1)
print('C-Hull Test Accuracy:', np.mean(pred_labels == true_labels))











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

plt.title('C-Hull Classifier: Separable Data (Train)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.axis('equal')
plt.tight_layout()










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

plt.title('C-Hull Classifier: Separable Data (Test)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()