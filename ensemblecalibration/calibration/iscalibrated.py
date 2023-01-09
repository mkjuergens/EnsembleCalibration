import numpy as np
from scipy.optimize import linprog

def isCalibrated(P, p):
    M = len(P)
    K = len(p)
    c = np.zeros(M)
    A = np.r_[P.T,np.ones((1,M))]
    b = np.r_[p, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    
    return lp.success

