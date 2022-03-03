from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from scipy.special import expit
import numpy as np

# get indicator vector from signed vector
def get_0_1(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else 0 for i in range(n)]).flatten()

# get signed vector from indicator vector
def get_sgn_0_1(vec):
    n = len(vec)
    return np.asarray([1 if vec[i] == 1 else -1 for i in range(n)]).flatten()

# conmpute bounds on Radon-Nikodym derivative
def get_bnds(gamma, p_s, n):
    gamma_inv = 1/gamma
    l_bnd = (1-p_s+(gamma*p_s))*gamma_inv
    u_bnd = gamma*(1-p_s)+p_s
    return [l_bnd * np.ones(n), u_bnd * np.ones(n)]

# estimate \pi^b, the behavior policy
def estimate_prop(x, T, predict_x, predict_T):
    clf_dropped = LogisticRegression()
    clf_dropped.fit(x, T)
    est_prop = clf_dropped.predict_proba(predict_x)
    est_Q = np.asarray( [est_prop[k,1] if predict_T[k] == 1 else est_prop[k,0] for k in range(len(predict_T))] )
    return [est_Q, clf_dropped]

# evaluate behavior policy on new data
def get_prop(clf, x, T):
    est_prop = clf_dropped.predict_proba(x)
    est_Q = np.asarray( [est_prop[k,1] if T[k] == 1 else est_prop[k,0] for k in range(len(T))] )
    return est_Q

# estimate outcome function $\mu_t
def estimate_outcome(x, T, Y, predict_x):
    clf_1 = KernelRidge(alpha=1.0)
    clf_0 = KernelRidge(alpha=1.0)
    clf_1.fit(x[T==1], Y[T==1])
    clf_0.fit(x[T==0], Y[T==0])
    mu_hat_1 = clf_1.predict(predict_x).reshape(-1,1)
    mu_hat_0 = clf_0.predict(predict_x).reshape(-1,1)
    return np.concatenate((mu_hat_0, mu_hat_1), axis=1)

# policy \pi
def logistic_pol_asgn(theta, x):
    ''' Requires an intercept term
    '''
    theta = theta.flatten()
    if len(theta) == 1:
        logit = np.multiply(x, theta).flatten()
    else:
        logit = np.dot(x, theta).flatten()
    
    LOGIT_TERM_POS = 1/(1+np.exp( -logit ))
    return LOGIT_TERM_POS

# for simulation study: sample from selection variables
def get_selection_var(X, r_seed, xi):
    np.random.seed(r_seed)
    c = cte(X, xi)
    P_S_X = 0.5+0.475*np.tanh(-c/0.1)
    S = np.random.binomial(1, P_S_X)
    return [S, P_S_X]

# for simulation study: treatment effect C(X)
def cte(X, xi): 
    n = X.shape[0]
    beta = np.array([-1.5, 1, -1.5, 1, 0.5])
    return np.dot(X, beta) + 2.5 + 2* (-2)*xi
    
# for simulation study: base effect m(X)
def m(X, xi):
    beta = np.array([0, 0.75, -0.5, 0, -1])
    w = 1
    return np.dot(X, beta) + w*xi - (-2)*xi

#----------------------------------------------- For Evaluation ---------------------------------------------------------
# for data-driven calibration of P(S=1)
def get_selection_prob_proxy(x_train, x_tar):
    n = x_train.shape[0]    
    N = x_tar.shape[0]
    
    #Naive proxy
    proxy_1 = n/N
    
    #Proxy as described in main paper
    ind = np.intersect1d(x_train, x_tar, return_indices=True)
    S = np.zeros(x_tar.shape)
    S[ind[2]] = 1

    clf_dropped = LogisticRegression()
    clf_dropped.fit(x_tar.reshape(-1,1), S)
    ps_x = clf_dropped.predict_proba(x_tar.reshape(-1,1))[:, 1]
    proxy_2 = np.mean(ps_x)
    return [proxy_1, proxy_2]
    
# for simulation study: calculate true gamma
def get_true_gamma(X, xi):
    ps_x = get_selection_var(X, 0, xi)[1]
    ps = np.mean(ps_x)
    u = max(ps/ps_x)
    gamma = (u - ps)/(1-ps)
    return [gamma, ps]


