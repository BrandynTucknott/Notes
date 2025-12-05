# projected gradient descent for C-hull classifier

# =================================================
# pseudo code:
#   In: data points A, B
#       step size alpha
#       max iterations MAX_ITER
#       tolerance TOL (for convergence)
#   Parameters: u, v
#   Out: C-hull classifier optimal u,v
# =================================================
'''
    initialize u,v
    initialize prev_obj (||Au - Bv||^2 at iter 0)
    for i in 1 to MAX_ITER do:
        diff = Au - Bv (compute objective function value at current iteration w/out norm squared)
        
        gu = 2 * A^T * diff (compute gradient w.r.t. u)
        gv = -2 * B^T * diff (compute gradient w.r.t. v)
        
        u_temp = u - alpha * gu (gradient descent step for u)
        v_temp = v - alpha * gv (gradient descent step for v)
        
        u = project_onto_simplex(u_temp) (project u back onto simplex)
        v = project_onto_simplex(v_temp) (project v back onto simplex)
        
        obj = norm(Au - Bv)^2 (compute objective function value)
        
        if |obj - prev_obj| < TOL (convergence check) then
            break
            
        prev_obj = obj (prepare for next iteration)
            
        at this point, current u,v are the optimal values (within tolerance)
'''


from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import time

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

# initialize parameters + plotting helpers
alpha = 0.01
MAX_ITER = 10_000
TOL = 1e-6
u = np.ones(A.shape[1]) / A.shape[1]
v = np.ones(B.shape[1]) / B.shape[1]
prev_obj = np.linalg.norm(A @ u - B @ v)**2

obj_history = [prev_obj]
time_history = [0]

for i in range(MAX_ITER):
    start = time.time()
    diff = A @ u - B @ v
    
    gu = 2 * A.T @ diff
    gv = -2 * B.T @ diff
    
    u_temp = u - alpha * gu
    v_temp = v - alpha * gv
    
    u = project_to_simplex(u_temp)
    v = project_to_simplex(v_temp)
    
    obj = np.linalg.norm(A @ u - B @ v)**2
    
    if abs(obj - prev_obj) < TOL:
        break
    
    prev_obj = obj
    end = time.time()
    
    obj_history.append(obj)
    time_history.append(1000 * (time_history[-1] + end - start)) # new total time = total run time + iteration time
    if i == MAX_ITER - 1:
        print("Reached maximum iterations without convergence.")
        
        
        
        
        
        
        
        
        
        
        
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

plt.title('C-Hull Classifier: Separable Data (Test)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
# plt.show()










# iteration vs objective value plot
plt.figure(figsize=(8,5))
plt.plot(obj_history)
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Iteration vs Objective Value (PGD C-Hull)")
plt.grid(True)
plt.tight_layout()
# plt.show()



# time vs objective value plot
plt.figure(figsize=(8,5))
plt.plot(time_history, obj_history)
plt.xlabel("Time (ms)")
plt.ylabel("Objective Value")
plt.title("Time vs Objective Value (PGD C-Hull)")
plt.grid(True)
plt.tight_layout()
plt.show()