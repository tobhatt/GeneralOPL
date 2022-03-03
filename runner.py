import numpy as np
import pickle
from Syn_data import *
from helper import *
from load_rct_data import *
from GenPol import *
from sklearn.preprocessing import MinMaxScaler
import argparse


parser = argparse.ArgumentParser()

# Parameters

parser.add_argument("--seed",
                    default=0,
                    type=int,
                    help="Random seed")
parser.add_argument("--policy_method",
                    default='DM',
                    type=str,
                    help="Policy method to be used")
parser.add_argument("--simulation",
                    default='y',
                    type=str,
                    help="y/n to decide if simulated data is used or not")
parser.add_argument("--max_iter",
                    default=2,
                    type=int,
                    help="Maximal number of iterations")
parser.add_argument("--gam",
                    default=1.0,
                    type=float,
                    help="Gamma for bounding odd-ratio")

args = parser.parse_args()


def gen_data_run_for_gamma(random_seed, GAMS, config_params):
    if config_params['simulation']:
        X, Y, T, xi = get_data_test(config_params['n_sample'], 0)
        S = get_selection_var(X, 0, xi)[0]
        x, y, t = get_data_train(X, Y, T, S)
        x = np.hstack([x, np.ones([x.shape[0],1])])
	
        ##Use same dataset as in main paper
        #data = np.load('sim_data.npy', allow_pickle=True)
        #X, Y, T , xi = data[0]
        #x, y, t = data[1]  
        #x = np.hstack([x, np.ones([x.shape[0],1])])
    
        #Proxy for P(S=1)
        x_train = x[:, 0]
        x_tar = X[:, 0] 
        p_s = get_selection_prob_proxy(x_train, x_tar)[1]
        
    else:
        x, y, t = sample_rct()
        x = np.hstack([x, np.ones([x.shape[0],1])])
        p_s = 0.5 #No covariate from target population available - conservative 
    
    #estimate \pi^b
    pol_b = estimate_prop(x, t, x, t)[0]
    
    #get true gamma if we run simulation
    if config_params['simulation']:
        t_gam, t_ps = get_true_gamma(X, xi)
    
    GenRobPols = GenPolicy(verbose = True, seed=random_seed, treatment_n = 'binary')
    GenRobPols.fit(x, t, y, pol_b, p_s, GAMS, config_params)
    
    if config_params['simulation']:
        test_data = {'x_test': X, 't_test': T, 'y_test': Y, 'xi':xi}
        GenRobPols.simulation_study_target_val(GAMS, test_data)
        GenRobPols.true_gamma = [t_gam, t_ps]
        
    return GenRobPols

    
if args.simulation=='y':
    simulation_flag=True
else:
    simulation_flag=False
GAMS = [args.gam]

config_params = {'max_iter': args.max_iter,
                 'tol': 1e-4,
                 'scal_val': 1.0,
                 'sc_lambda': 0.001,
                 'theta_lb': -10000.,
                 'theta_ub': 10000.,
                 'policy_method': args.policy_method,
                 'algorithm': 'L-BFGS-B',
                 'simulation': simulation_flag,
                 'n_sample': 6000} 

res = [gen_data_run_for_gamma(args.seed, GAMS, config_params)]

if simulation_flag:
    pickle.dump(res, open('Experiments/res-'+ 'simulation_' + config_params['policy_method'] +'_'+str(args.seed)+'_'+ str(args.gam) +'.pkl', 'wb'))
else:
    pickle.dump(res, open('Experiments/res-'+ 'clinical_trial_' + config_params['policy_method'] +'_'+str(args.seed)+'_'+ str(args.gam) +'.pkl', 'wb'))




