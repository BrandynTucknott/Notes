import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.io import loadmat

# -------------------------
# Simplex projection (uncapped and capped)
# -------------------------
def simplex(v, d=1.0, tol=1e-12, max_iter=200):
    """
    Project v onto the capped simplex:
        { w : sum(w) = 1, 0 <= w_i <= d }
    If d == 1.0 this reduces to the standard simplex projection (uncapped).
    Uses analytic projection for d==1 and a robust bisection for capped case.
    """
    v = np.asarray(v, dtype=float).copy()
    n = v.size

    if d <= 0:
        raise ValueError("d must be positive")

    # standard (uncapped) simplex projection (Duchi et al.)
    if abs(d - 1.0) < 1e-15:
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0].max()
        theta = (cssv[rho] - 1) / (rho + 1)
        w = np.maximum(v - theta, 0.0)
        return w

    # capped simplex: solve for theta such that sum(clamp(v - theta, 0, d)) == 1
    # monotone in theta so we can bisection.
    # quick feasibility check: if sum(clamp(v,0,d)) == 1, return clipped v
    clipped = np.clip(v, 0.0, d)
    s = clipped.sum()
    if abs(s - 1.0) < tol:
        return clipped

    # bisection bounds for theta
    # Lower bound: make all entries as large as possible => theta_low = min(v) - d
    # Upper bound: make all entries as small as possible => theta_high = max(v)
    theta_low = np.min(v) - d - 1.0
    theta_high = np.max(v) + 1.0

    for _ in range(max_iter):
        theta = 0.5 * (theta_low + theta_high)
        w = np.clip(v - theta, 0.0, d)
        s = w.sum()
        if abs(s - 1.0) <= tol:
            return w
        if s > 1.0:
            theta_low = theta
        else:
            theta_high = theta
    w = np.clip(v - theta, 0.0, d)
    if w.sum() > 0:
        w = w / w.sum()
    return w

# =========================
# Load data (adjust path if needed)
# =========================
data = loadmat('code/school_work/ECE_569/HW_3/train_separable.mat')
A = data['A']
B = data['B']

# =========================
# Helpers & initialization
# =========================
def obj_val(u, v):
    r = A @ u - B @ v
    return float(np.linalg.norm(r)**2)

m = A.shape[1]
n = B.shape[1]

# compute a safe step size alpha = 1 / L where L = 2*(||A||^2 + ||B||^2)
normA2 = np.linalg.norm(A, 2)**2
normB2 = np.linalg.norm(B, 2)**2
L = 2.0 * (normA2 + normB2)
alpha_default = 1.0 / L

print(f"Spectral-norm^2(A)={normA2:.4g}, ^2(B)={normB2:.4g}, L={L:.4g}, alpha={alpha_default:.4g}")

# containers
u0 = np.ones(m) / m
v0 = np.ones(n) / n

pgd_chull_iter = [obj_val(u0, v0)]
nag_chull_iter = [obj_val(u0, v0)]
admm_chull_iter = [obj_val(u0, v0)]
pgd_chull_time = [0.0]
nag_chull_time = [0.0]
admm_chull_time = [0.0]

# =========================
# PGD
# =========================
alpha = alpha_default
MAX_ITER_PGD = 1000
TOL = 1e-8
u = u0.copy()
v = v0.copy()
prev_obj = obj_val(u, v)

for i in range(MAX_ITER_PGD):
    t0 = time.time()
    diff = A @ u - B @ v
    gu = 2.0 * A.T @ diff
    gv = -2.0 * B.T @ diff

    u_temp = u - alpha * gu
    v_temp = v - alpha * gv

    u = simplex(u_temp)
    v = simplex(v_temp)

    obj = obj_val(u, v)
    t1 = time.time()

    pgd_chull_iter.append(obj)
    pgd_chull_time.append(pgd_chull_time[-1] + 1000.0 * (t1 - t0))

    if abs(obj - prev_obj) < TOL:
        print(f'PGD converged in {i+1} iterations, obj={obj:.6e}')
        break
    prev_obj = obj
else:
    print("PGD: Reached maximum iterations without convergence.")

# =========================
# Nesterov (FISTA-style)
# =========================
alpha = alpha_default
MAX_ITER_NAG = 1000
u = u0.copy()
prev_u = u.copy()
v = v0.copy()
prev_v = v.copy()


prev_t = 1.0
prev_obj = obj_val(u, v)

for i in range(MAX_ITER_NAG):
    t0 = time.time()

    t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * (prev_t**2)))
    beta = (prev_t - 1.0) / t

    y_u = u + beta * (u - prev_u)
    y_v = v + beta * (v - prev_v)

    diff = A @ y_u - B @ y_v
    gu = 2.0 * A.T @ diff
    gv = -2.0 * B.T @ diff

    u_temp = y_u - alpha * gu
    v_temp = y_v - alpha * gv

    u_new = simplex(u_temp)
    v_new = simplex(v_temp)

    prev_u, u = u, u_new
    prev_v, v = v, v_new

    obj = obj_val(u, v)
    t1 = time.time()

    nag_chull_iter.append(obj)
    nag_chull_time.append(nag_chull_time[-1] + 1000.0 * (t1 - t0))

    prev_t = t

    if abs(obj - prev_obj) < TOL:
        print(f'Nesterov (FISTA) converged in {i+1} iterations, obj={obj:.6e}')
        break
    prev_obj = obj
else:
    print("Nesterov: Reached maximum iterations without convergence.")

print("PGD last objs (first 10):", pgd_chull_iter[:10])
print("NAG last objs (first 10):", nag_chull_iter[:10])

# =========================
# ADMM
# =========================
u = u0.copy()
x = u.copy()
v = v0.copy()
y = v.copy()

rho = 50.0
tau_inc = 2.0
tau_dec = 2.0
mu = 10.0
TOL_ADMM = 1e-8
MAX_ITER_ADMM = 500

lambda_u = np.zeros(m)
lambda_v = np.zeros(n)

AtA = A.T @ A
BtB = B.T @ B
AtB = A.T @ B
BtA = B.T @ A

def chol_solve(L, b):
    # L is chol factor of matrix M = L @ L.T
    tmp = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, tmp)

Lu_factor = np.linalg.cholesky(2.0 * AtA + rho * np.eye(m))
Lv_factor = np.linalg.cholesky(2.0 * BtB + rho * np.eye(n))

for i in range(MAX_ITER_ADMM):
    t0 = time.time()

    rhs_u = 2.0 * AtB @ v + rho * (x - lambda_u)
    u = chol_solve(Lu_factor, rhs_u)

    rhs_v = 2.0 * BtA @ u + rho * (y - lambda_v)
    v = chol_solve(Lv_factor, rhs_v)

    x_old = x.copy()
    y_old = y.copy()

    # project onto uncapped simplex (sum==1, 0<=.)
    x = simplex(u + lambda_u)   # default d=1 -> uncapped simplex
    y = simplex(v + lambda_v)

    lambda_u += (u - x)
    lambda_v += (v - y)

    prim = np.sqrt(np.linalg.norm(u - x)**2 + np.linalg.norm(v - y)**2)
    dual = rho * np.sqrt(np.linalg.norm(x - x_old)**2 + np.linalg.norm(y - y_old)**2)

    rho_old = rho
    if prim > mu * dual:
        rho *= tau_inc
    elif dual > mu * prim:
        rho /= tau_dec

    if rho != rho_old:
        scale = rho_old / rho
        lambda_u *= scale
        lambda_v *= scale
        Lu_factor = np.linalg.cholesky(2.0 * AtA + rho * np.eye(m))
        Lv_factor = np.linalg.cholesky(2.0 * BtB + rho * np.eye(n))

    t1 = time.time()

    admm_chull_iter.append(obj_val(u, v))
    admm_chull_time.append(admm_chull_time[-1] + 1000.0 * (t1 - t0))

    if prim < TOL_ADMM and dual < TOL_ADMM:
        print(f"ADMM converged in {i+1} iterations, obj={admm_chull_iter[-1]:.6e}")
        break
else:
    print("ADMM: Reached maximum iterations without convergence.")

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(8,5))
plt.plot(pgd_chull_iter, label='PGD')
plt.plot(nag_chull_iter, label='Nesterov (FISTA)')
plt.plot(admm_chull_iter, label='ADMM')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Iteration vs Objective Value (C-Hull)")
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.figure(figsize=(8,5))
plt.plot(pgd_chull_time, pgd_chull_iter, label='PGD')
plt.plot(nag_chull_time, nag_chull_iter, label='Nesterov (FISTA)')
plt.plot(admm_chull_time, admm_chull_iter, label='ADMM')
plt.xlabel("Time (ms)")
plt.ylabel("Objective Value")
plt.title("Time vs Objective Value (C-Hull)")
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()





# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from scipy.io import loadmat

# '''
# Projection of vector v onto simplex (this function is not my code, but copied from somwhere online)
# Based on the algorithm by Duchi et al. (2008).
# sum(w) = 1
# 0 <= w_i <= d
# by default is an uncapped simplex
# '''
# def simplex(v, d=1):
#     def project_to_simplex(v):
#         v = np.asarray(v)
#         n = v.size
#         u = np.sort(v)[::-1]
#         cssv = np.cumsum(u)
#         rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0].max()
#         theta = (cssv[rho] - 1) / (rho + 1)
#         w = np.maximum(v - theta, 0)
#         return w
    
    
#     if d == 1:
#         return project_to_simplex(v)
#     tol = 1e-12
#     v = np.asarray(v).copy()
#     n = len(v)
    
#     # Clip initial guess to [0, d]
#     v = np.clip(v, 0, d)
    
#     # Check if already feasible
#     s = v.sum()
#     if abs(s - 1.0) < tol:
#         return v

#     # Sort descending
#     u = np.sort(v)[::-1]

#     # Prefix sum of sorted vector
#     cssv = np.cumsum(u)
    
#     # Find the threshold index
#     rho = -1
#     for j in range(n):
#         t = (cssv[j] - 1.0) / (j + 1)
#         if u[j] - t > d:
#             continue  # capped by d
#         elif u[j] - t > 0:
#             rho = j
#             break

#     if rho == -1:
#         # If no active set found, just clip uniformly
#         w = np.clip(v + (1 - v.sum()) / n, 0, d)
#         return w

#     theta = (cssv[rho] - 1.0) / (rho + 1)
    
#     # Final projection
#     w = np.clip(v - theta, 0, d)
    
#     # Fix tiny numerical issues
#     w = w / w.sum()
#     return w




# data = loadmat('code/school_work/ECE_569/HW_3/train_separable.mat')
# A = data['A']
# B = data['B']
# u = np.ones(A.shape[1]) / A.shape[1]
# v = np.ones(B.shape[1]) / B.shape[1]
# # all objective function values at each iteration
# pgd_chull_iter = [np.linalg.norm(A @ u - B @ v)**2]
# nag_chull_iter = [np.linalg.norm(A @ u - B @ v)**2]
# admm_chull_iter = [np.linalg.norm(A @ u - B @ v)**2]

# # all objective function values at each time step
# pgd_chull_time = [0]
# nag_chull_time = [0]
# admm_chull_time = [0]



# # pgd_chull
# alpha = 0.01
# MAX_ITER = 10
# TOL = 1e-6
# u = np.ones(A.shape[1]) / A.shape[1]
# v = np.ones(B.shape[1]) / B.shape[1]
# prev_obj = np.linalg.norm(A @ u - B @ v)**2

# for i in range(MAX_ITER):
#     start = time.time()
#     diff = A @ u - B @ v
    
#     gu = 2 * A.T @ diff
#     gv = -2 * B.T @ diff
    
#     u_temp = u - alpha * gu
#     v_temp = v - alpha * gv
    
#     u = simplex(u_temp)
#     v = simplex(v_temp)
    
#     obj = np.linalg.norm(A @ u - B @ v)**2
    
#     end = time.time()
    
#     pgd_chull_iter.append(obj)
#     pgd_chull_time.append(pgd_chull_time[-1] + 1000 * (end - start))
    
#     if abs(obj - prev_obj) < TOL:
#         print(f'converged in {i + 1} iterations.')
#         break
#     prev_obj = obj
#     if i == MAX_ITER - 1:
#         print("Reached maximum iterations without convergence.")
        
        
        
        
        
# # nesterov chull
# alpha = 0.01
# MAX_ITER = 10
# TOL = 1e-6
# u = np.ones(A.shape[1]) / A.shape[1]
# prev_u = np.ones(A.shape[1]) / A.shape[1]
# v = np.ones(B.shape[1]) / B.shape[1]
# prev_v = np.ones(B.shape[1]) / B.shape[1]

# prev_t = 0
# prev_obj = np.linalg.norm(A @ u - B @ v)**2

# for i in range(MAX_ITER):
#     start = time.time()
#     t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * prev_t**2))
#     beta = (prev_t + 1.0) / t
    
#     y_u = u + beta * (u - prev_u)
#     y_v = v + beta * (v - prev_v)
    
#     diff = A @ y_u - B @ y_v
#     gu = 2 * A.T @ diff
#     gv = -2 * B.T @ diff
    
#     u_temp = y_u - alpha * gu
#     v_temp = y_v - alpha * gv
    
#     u_new = simplex(u_temp)
#     v_new = simplex(v_temp)
    
#     prev_u, u = u, u_new
#     prev_v, v = v, v_new
    
#     obj = np.linalg.norm(A @ u - B @ v)**2
    
#     end = time.time()
    
#     nag_chull_iter.append(obj)
#     nag_chull_time.append(nag_chull_time[-1] + 1000 * end - 1000 * start) # new total time = total run time + iteration time
    
#     if abs(obj - prev_obj) < TOL:
#         print(f'converged in {i + 1} iterations.')
#         break
#     prev_obj = obj
#     if i == MAX_ITER - 1:
#         print("Reached maximum iterations without convergence.")
        
        
# print(pgd_chull_iter)
# print(nag_chull_iter)
        
# # admm chull
# # --- initialization ---
# u = np.ones(A.shape[1]) / A.shape[1]
# x = np.ones(A.shape[1]) / A.shape[1]
# v = np.ones(B.shape[1]) / B.shape[1]
# y = np.ones(B.shape[1]) / B.shape[1]

# # parameters + limiters
# rho = 50.0
# tau_inc = 2.0
# tau_dec = 2.0
# mu = 10.0
# TOL = 1e-6
# MAX_ITER = 10

# # scaled duals
# lambda_u = np.zeros(A.shape[1])
# lambda_v = np.zeros(B.shape[1])

# # precompute blocks
# AtA = A.T @ A
# BtB = B.T @ B
# AtB = A.T @ B
# BtA = B.T @ A

# def chol_solve(L, b):
#     """solve (L L^T) x = b efficiently"""
#     y = np.linalg.solve(L, b)
#     return np.linalg.solve(L.T, y)

# # initial Cholesky factors (for current rho)
# Lu_factor = np.linalg.cholesky(2 * AtA + rho * np.eye(A.shape[1]))
# Lv_factor = np.linalg.cholesky(2 * BtB + rho * np.eye(B.shape[1]))

# for i in range(MAX_ITER):
#     t0 = time.time()
#     # u update: solve (2 A^T A + rho I) u = 2 A^T B v + rho (x - lambda_u)
#     rhs_u = 2 * AtB @ v + rho * (x - lambda_u)
#     u = chol_solve(Lu_factor, rhs_u)

#     # v update
#     rhs_v = 2 * BtA @ u + rho * (y - lambda_v)
#     v = chol_solve(Lv_factor, rhs_v)

#     # store old simplex projections for dual residual (and for potential rho-adapt)
#     x_old = x.copy()
#     y_old = y.copy()

#     # projection step (replace with your projection function)
#     x = simplex(u + lambda_u)
#     y = simplex(v + lambda_v)

#     # scaled dual updates
#     lambda_u += (u - x)
#     lambda_v += (v - y)

#     # primal and dual residuals (scaled form)
#     prim = np.sqrt(np.linalg.norm(u - x)**2 + np.linalg.norm(v - y)**2)
#     dual = rho * np.sqrt(np.linalg.norm(x - x_old)**2 + np.linalg.norm(y - y_old)**2)

#     # ADAPTIVE RHO (rescale duals if rho changes)
#     rho_old = rho
#     if prim > mu * dual:
#         rho *= tau_inc
#     elif dual > mu * prim:
#         rho /= tau_dec

#     if rho != rho_old:
#         # rescale the scaled dual variables when rho changes
#         scale = rho_old / rho
#         lambda_u *= scale
#         lambda_v *= scale

#         # recompute Cholesky factors for new rho
#         Lu_factor = np.linalg.cholesky(2 * AtA + rho * np.eye(A.shape[1]))
#         Lv_factor = np.linalg.cholesky(2 * BtB + rho * np.eye(B.shape[1]))
    
#     t1 = time.time()
#     admm_chull_iter.append(float((A @ u - B @ v) @ (A @ u - B @ v)))
#     admm_chull_time.append(admm_chull_time[-1] + 1000 * (end - start))

#     # stopping (you can replace with Boyd eps_pri/eps_dual if you want)
#     if prim < TOL and dual < TOL:
#         print(f"Converged in {i+1} iterations.")
#         break
# else:
#     print("Reached maximum iterations without convergence.")


    
    
    
    
    
    
    
    
    
    
    
        





# # iteration vs objective value plot
# plt.figure(figsize=(8,5))
# plt.plot(pgd_chull_iter, label='PGD', color='blue')
# plt.plot(nag_chull_iter, label='Nesterov', color='red')
# plt.plot(admm_chull_iter, label='ADMM', color='green')
# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("Iteration vs Objective Value (C-Hull)")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()



# # time vs objective value plot
# plt.figure(figsize=(8,5))
# plt.plot(pgd_chull_time, pgd_chull_iter, label='PGD', color='blue')
# plt.plot(nag_chull_time, nag_chull_iter, label='Nesterov', color='red')
# plt.plot(admm_chull_time, admm_chull_iter, label='ADMM', color='green')
# plt.xlabel("Time (ms)")
# plt.ylabel("Objective Value")
# plt.title("Time vs Objective Value (C-Hull)")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()