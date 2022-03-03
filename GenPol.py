from helper import *
from MMPCCP import *
import scipy.special
from datetime import datetime
import numpy as np

class GenPolicy:
    def __init__(self, verbose = False, seed=0, treatment_n='binary'):
        
        self.verbose = verbose
        self.treatment_n = treatment_n
        self.seed = seed

        # store training data when passed in by .fit
        self.x = None
        self.t = None
        self.y = None
        self.mu_hat = None
        self.pol_b = None
        self.p_s = None
        self.pols_dict = None
        self.pol_val_dict = None
        self.method = None
        
        # for simulation study
        self.tar_val = None
        self.tar_data = None
        self.true_gamma = None
        
    def fit(self, x, t, y, pol_b, p_s, GAMS, config_params):
        if self.treatment_n == 'binary':
            t = get_0_1(t) # make sure treatment is 0-1
        else:
            raise ValueError("Only implemented for binary decisions.")
        
        np.random.seed(self.seed)
        
        # we input data with intercept
        opt_params = {'x': x,
                      't':t,
                      'y':y,
                      'pol_b': pol_b,
                      'p_s': p_s,
                      'verbose': self.verbose,
                      'theta_lb': config_params['theta_lb'],
                      'theta_ub': config_params['theta_ub'],
                      'scal_val': config_params['scal_val'],
                      'sc_lambda': config_params['sc_lambda'],
                      'algorithm': config_params['algorithm'],
                      'policy_method': config_params['policy_method'],
                      'max_iter': config_params['max_iter'],
                      'tol': config_params['tol']}
        self.x = x; self.t = t; self.y = y; self.pol_b = pol_b; self.p_s = p_s;self.method = config_params['policy_method']
        
        print(config_params['policy_method'])
        print(" ")

        if config_params['policy_method'] != 'IPW':
            mu_hat = estimate_outcome(x, t, y, x)
            self.mu_hat = mu_hat
            opt_params.update({'mu_hat': mu_hat})
            
        pols = [ None ] * len(GAMS)
        pol_vals = np.zeros(len(GAMS))

        # iterate over values of GAMMAS and train
        for ind_g, gam in enumerate(GAMS):
            print('Current gamma: ', gam)
            n = x.shape[0]
            bnds = get_bnds(gam, p_s, n)
            print('Current bounds: [', bnds[0][0], ',', bnds[1][0], ']')
            # Compute constant term
            help_mat=np.zeros((n+1,n))
            for j in range(n+1):
                if j>0:
                    help_mat[j,:]=np.hstack([np.ones(j)*bnds[1][0],np.ones(n-j)*bnds[0][0]])
                else:
                    help_mat[j,:]=np.ones(n-j)*bnds[0][0]
            
            for j in range(n+1):
                help_mat[j,:]=help_mat[j,:]/np.sum(help_mat[j,:])
            
            const_term=help_mat.sum(axis=0)
            
            opt_params.update({'l_bnd': bnds[0],'u_bnd': bnds[1],'const_term': const_term})
            
            theta_zero = np.random.normal(0, 0.1, x.shape[1])
            
            now = datetime.now()
            [gen_theta, pol_val] = MMPCCP(theta_zero, opt_params)
            if self.verbose:
                print('CPU Time: ',datetime.now() - now)
            
            pols[ind_g] = gen_theta
            pol_vals[ind_g] = pol_val
        
        # After fitting: class contains a list of policies
        self.pols_dict = dict(zip(GAMS, pols))
        self.pol_val_dict = dict(zip(GAMS, pol_vals))

    def predict(self, x, gamma):
        theta = self.pols_dict[gamma]
        return logistic_pol_asgn(theta,x)

    def simulation_study_target_val(self, GAMS, test_data):
        TAR_VAL = [ None ] * len(GAMS)
        x = test_data['x_test']
        xi = test_data['xi']
        CTE = cte(x, xi)
        Y_0 = m(x, xi)
        x_aug = np.hstack([x, np.ones([x.shape[0],1])])
        for ind_g, gam in enumerate(GAMS):
            pol_prob = self.predict(x_aug, gam)     
            TAR_VAL[ind_g] = (pol_prob * CTE + Y_0).mean()
        self.tar_val = dict(zip(GAMS, TAR_VAL))
        self.tar_data = test_data