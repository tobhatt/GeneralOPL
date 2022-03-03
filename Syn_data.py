import numpy as np
from scipy.special import expit
import matplotlib as plt
from helper import *

def get_oracle_policy_value(X, xi, n_gam):
    Y_0 = m(X, xi)
    eff = cte(X, xi)
    n=eff.shape[0]
    pol = np.array([1 if eff[i] <= 0 else 0 for i in range(n)])
    pol_val = pol * eff + Y_0
    return np.repeat(pol_val.mean(), n_gam)

def get_data_test(n, r_seed):
    np.random.seed(r_seed)
    #Sample from test distribution
    mean = [-1, 0.5, -1, 0, -1]
    cov = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    X = np.random.multivariate_normal(mean, cov, n)
    T = np.random.binomial(1, 0.5, n)
    xi = np.random.binomial(1, 0.5, n)
    C =  cte(X, xi)
    Y_0 = m(X, xi) + 0.*C + np.random.normal(np.zeros(n), 1.)
    Y_1 = m(X, xi) + 1.*C + np.random.normal(np.zeros(n), 1.)
    Y = np.array([Y_1[i] if T[i]==1 else Y_0[i] for i in range(n)])  
    return(X, Y, T, xi)

def get_data_train(X, Y, T, S):
    n = X.shape[0]
    #Select training dataset
    idx_train = np.array(range(0, n))[(S==1)]
    return(X[idx_train], Y[idx_train], T[idx_train])
    
