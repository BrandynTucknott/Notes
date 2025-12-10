import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

"""
First experiment generated a matrix M = M_L @ M_R where M_L, M_R have i.i.d gaussian entries
Parameters:
    n - size of matrix M
    r - rank of M; M_L is n x r and M_R is r x n
    m - number of samples
    tolerance - a solution X is considered recovered if ||X - M|| / ||M|| < tolerance where ||*|| is the frobenius norm
"""
def experimentGaussian(n, r, m, tolerance=1e-3):
    # generate M
    M_L = np.array([[np.random.normal() for _ in range(r)] for _ in range(n)])
    M_R = np.array([[np.random.normal() for _ in range(n)] for _ in range(r)])
    M = M_L @ M_R
    M_F = np.linalg.norm(M, ord='fro') # frobenius norm of M
    
    # generate m unif. rand. observations
    
    
    # variables, constraints, and objective function
    # tolerance is incorporated into our constraint
    X = cp.Variable(n, n)
    constraints = [
        cp.norm(X - M, 'fro') / M_F <= tolerance,
        X
    ]
    obj = cp.Minimize(cp.normNuc(X))
    
    # solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    
"""
First experiment generated a matrix M = M_L @ M_R where M_L, M_R have i.i.d gaussian entries
Parameters:
    n - size of matrix M
    r - rank of M; M_L is n x r and M_R is r x n
    m - number of samples
    tolerance - a solution X is considered recovered if ||X - M|| / ||M|| < tolerance where ||*|| is the frobenius norm
"""
def experimentPSD(tolerance=1e-3):
    ...
    
"""
First experiment generated a matrix M = M_L @ M_R where M_L, M_R have i.i.d gaussian entries
Parameters:
    n - size of matrix M
    r - rank of M; M_L is n x r and M_R is r x n
    m - number of samples
    tolerance - a solution X is considered recovered if ||X - M|| / ||M|| < tolerance where ||*|| is the frobenius norm
"""
def experimentLinearMap(tolerance=1e-3):
    ...
    

















def main():
    

if __name__ == "__main__":
    main()