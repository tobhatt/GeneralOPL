import numpy as np
from helper import *
import scipy
from scipy.optimize import minimize
from numpy import linalg as LA

def policy(theta,x):
    return logistic_pol_asgn(theta,x)
      
def tilde_g(theta,x,sc_lambda):
    lin_val=np.dot(theta,x)
    if lin_val>=0:
        val=0.25*lin_val+0.5+0.5*sc_lambda*lin_val**2
    else:
        val=0.5*np.tanh(0.5*lin_val)+0.5+0.5*sc_lambda*lin_val**2
    return val
    
def tilde_g_prime(theta,x,sc_lambda):
    lin_val=np.dot(theta,x)
    if lin_val>=0:
        g=0.25*x+lin_val*sc_lambda*x
    else:
        g=0.25*(1-np.square(np.tanh(0.5*lin_val)))*x+lin_val*sc_lambda*x
    return g
    
def tilde_h(theta,x,sc_lambda):
    lin_val=np.dot(theta,x)
    if lin_val>=0:
        val=0.25*lin_val-0.5*np.tanh(0.5*lin_val)+0.5*sc_lambda*lin_val**2
    else:
        val=0.0+0.5*sc_lambda*lin_val**2
    return val
    
def tilde_h_prime(theta,x,sc_lambda):
    lin_val=np.dot(theta,x)
    if lin_val>=0:
        g=(0.25-0.25*(1-np.square(np.tanh(0.5*lin_val))))*x+sc_lambda*lin_val*x
    else:
        g=0.0*x+sc_lambda*lin_val*x
    return g

def get_psi(theta,opt_params):
    
    x=opt_params['x']
    T=opt_params['t']
    y=opt_params['y']
    pol_b=opt_params['pol_b']
    mu_hat=opt_params['mu_hat']
    
    n = x.shape[0]
    
    if opt_params['policy_method'] == 'DM':
        pi_1 = policy(theta, x).flatten()
        
        return pi_1 * mu_hat[:, 1] + (1-pi_1) * mu_hat[:, 0]
    
    if opt_params['policy_method'] == 'NIPW':
        W_IPW = 1/pol_b
        C_IPW = (2*W_IPW)/(np.mean(W_IPW)) #Normalized inverse prob weights
        pi_1 = policy(theta, x).flatten()
        psi = np.array([pi_1[k]*C_IPW[k]*y[k] if T[k] == 1 else (1-pi_1[k])*C_IPW[k]*y[k] for k in range(n)])
        
        return psi
    
    if opt_params['policy_method'] == 'DR':        
        W_IPW = 1/pol_b
        
        pi_1 = policy(theta, x).flatten()
        direct_part = pi_1 * mu_hat[:, 1] + (1-pi_1) * mu_hat[:, 0]
        
        mu_T = np.array([mu_hat[k, 0]if T[k]==0 else mu_hat[k, 1] for k in range(n)])
        bias = y-mu_T
        bias_part = np.array([bias[k]*W_IPW[k]*pi_1[k] if T[k] == 1 else bias[k]*W_IPW[k]*(1-pi_1[k]) for k in range(n)])
        
        return direct_part + bias_part
    
def get_psi_prime(theta,i,opt_params):
    
    if opt_params['policy_method'] == 'DM':
        
        mu_hat_1=opt_params['mu_hat'][i,1]
        mu_hat_0=opt_params['mu_hat'][i,0]
        x=opt_params['x'][i,:]
        sig_prime=policy(theta,x)*(1-policy(theta,x))
        
        return (mu_hat_1-mu_hat_0)*sig_prime*x
    
    if opt_params['policy_method'] == 'NIPW':
        pol_b=opt_params['pol_b']
        x=opt_params['x'][i,:]
        T=opt_params['t'][i]
        y=opt_params['y'][i]
        
        W_IPW = 1/pol_b
        C_IPW = (2*W_IPW[i])/(np.mean(W_IPW)) #Normalized inverse prob weights
        sig_prime=policy(theta,x)*(1-policy(theta,x))
        
        return -C_IPW*(1-2*T)*y*sig_prime*x
    
    if opt_params['policy_method'] == 'DR':        
        pol_b=opt_params['pol_b']
        W_IPW = 1/pol_b[i]
        mu_hat=opt_params['mu_hat'][i]
        x=opt_params['x'][i,:]
        y=opt_params['y'][i]
        T=opt_params['t'][i]
        
        sig_prime=policy(theta,x)*(1-policy(theta,x))
        helpval=(1-2*T)*(y-opt_params['mu_hat'][i,T])
        
        return (mu_hat[1]-mu_hat[0]-W_IPW*helpval)*sig_prime*x

def h_i(theta,i,opt_params,grad_flag):
    
    sc_lambda=opt_params['sc_lambda']
    
    if opt_params['policy_method'] == 'DM':
        mu_hat_1=opt_params['mu_hat'][i,1]
        mu_hat_0=opt_params['mu_hat'][i,0]
        
        if grad_flag:
            if mu_hat_0>=0 and mu_hat_1>=0:
                g=tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1>=0:
                g=tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
            if mu_hat_0>=0 and mu_hat_1<0:
                g=tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1<0:
                g=tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
            return g
        else:            
            if mu_hat_0>=0 and mu_hat_1>=0:
                val=tilde_h(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_g(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1>=0:
                val=tilde_h(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_h(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)            
            if mu_hat_0>=0 and mu_hat_1<0:
                val=tilde_g(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_g(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1<0:
                val=tilde_g(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_h(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
            return val
    
    if opt_params['policy_method'] == 'NIPW':
        W_IPW = 1/opt_params['pol_b']
        C_IPW = (2*W_IPW)/(np.mean(W_IPW)) #Normalized inverse prob weights
        y=opt_params['y'][i]
        T=opt_params['t'][i]
        
        if grad_flag:
            if T==1:
                g=C_IPW[i]*y*tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)
            else:
                g=C_IPW[i]*y*tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)            
            return g
        else:
            if T==1:
                val=C_IPW[i]*y*tilde_h(theta,opt_params['x'][i,:],sc_lambda)
            else:
                val=C_IPW[i]*y*tilde_g(theta,opt_params['x'][i,:],sc_lambda)
            return val
            
    if opt_params['policy_method'] == 'DR':
        
        mu_hat_1=opt_params['mu_hat'][i,1]
        mu_hat_0=opt_params['mu_hat'][i,0]
        
        if grad_flag:
            if mu_hat_0>=0 and mu_hat_1>=0:
                g=tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1>=0:
                g=tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
            if mu_hat_0>=0 and mu_hat_1<0:
                g=tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1<0:
                g=tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
        else:            
            if mu_hat_0>=0 and mu_hat_1>=0:
                val=tilde_h(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_g(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1>=0:
                val=tilde_h(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_1+tilde_h(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)            
            if mu_hat_0>=0 and mu_hat_1<0:
                val=tilde_g(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_g(theta,opt_params['x'][i,:],sc_lambda)*mu_hat_0
            if mu_hat_0<0 and mu_hat_1<0:
                val=tilde_g(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_1)+tilde_h(theta,opt_params['x'][i,:],sc_lambda)*abs(mu_hat_0)
        
        W_IPW = 1/opt_params['pol_b']
        y=opt_params['y'][i]
        T=opt_params['t'][i]
        helpval=(1-2*T)*(y-opt_params['mu_hat'][i,T])
        
        if grad_flag:
            if helpval>=0:
                g+=W_IPW[i]*helpval*tilde_g_prime(theta,opt_params['x'][i,:],sc_lambda)
            else:
                g+=W_IPW[i]*abs(helpval)*tilde_h_prime(theta,opt_params['x'][i,:],sc_lambda)
            return g
        else:
            if helpval>=0:
                val+=W_IPW[i]*helpval*tilde_g(theta,opt_params['x'][i,:],sc_lambda)
            else:
                val+=W_IPW[i]*abs(helpval)*tilde_h(theta,opt_params['x'][i,:],sc_lambda)
            return val

def invert_permutation(permutation):
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=inv.dtype)
    return inv

def find_opt_weights(psi, opt_params):
    
    l_=opt_params['l_bnd']
    u_=opt_params['u_bnd']
    n=psi.shape[0]
    psi_sort_ind=np.argsort(psi)
    psi_sort=psi[psi_sort_ind]
    
    for k in range(n+1):
        lambda_k=(np.sum(psi_sort[:k])*l_[0]+np.sum(psi_sort[k:])*u_[0])/(k*l_[0]+(n-k)*u_[0])
        if lambda_k<psi_sort[k]:
            k_star=k
            break
    
    r_sort=np.hstack([np.ones(k_star)*l_[0],np.ones(n-k_star)*u_[0]])
    
    return lambda_k, r_sort[invert_permutation(psi_sort_ind)]

def CCP_objective(theta,opt_params,current_theta,scal_val):
    
    n = opt_params['x'].shape[0]
    psi = get_psi(theta,opt_params)
    val, r = find_opt_weights(psi,opt_params)
    
    his=list(map(lambda i: h_i(theta,i,opt_params,False), range(n)))
    grad_ks=list(map(lambda i: h_i(current_theta,i,opt_params,True), range(n)))
    grads=list(map(lambda i: h_i(theta,i,opt_params,True), range(n)))
    psi_primes=list(map(lambda i: get_psi_prime(theta,i,opt_params), range(n)))

    for i,hi in enumerate(his):
        grad_k=grad_ks[i]
        grad=grads[i]
        psi_prime=psi_primes[i]
                
        val+=opt_params['const_term'][i]*hi
        val-=opt_params['const_term'][i]*np.dot(theta,grad_k)
        
        if i==0:
            g1=r[i]*psi_prime
            g2=opt_params['const_term'][i]*(grad-grad_k)
        else:
            g1+=r[i]*psi_prime
            g2+=opt_params['const_term'][i]*(grad-grad_k)

    val=val*scal_val
    g1/=np.sum(r)
    g=g1+g2
    g=g*scal_val

    return (val,g)

def MMPCCP(theta_zero,opt_params):
    
    bds=scipy.optimize.Bounds(opt_params['theta_lb'], opt_params['theta_ub'], keep_feasible=False)
    
    current_theta=theta_zero
    old_theta=theta_zero+np.ones_like(theta_zero)*opt_params['tol']; # just for init
    iteration_counter=0
    current_fval=np.inf
    scal_val=opt_params['scal_val']
    
    while iteration_counter<opt_params['max_iter'] and LA.norm(old_theta-current_theta)>opt_params['tol']:
        
        old_theta=current_theta
        
        if opt_params['verbose']:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print('Iteration: ',iteration_counter)
            if np.isinf(opt_params['theta_lb']) and np.isinf(opt_params['theta_ub']):
                res=minimize(CCP_objective, current_theta, args=(opt_params,current_theta,scal_val), method=opt_params['algorithm'], options={'disp': False, 'gtol': 1e-6}, jac=True)
            else:
                res=minimize(CCP_objective, current_theta, args=(opt_params,current_theta,scal_val), method=opt_params['algorithm'], options={'disp': False, 'gtol': 1e-6}, bounds=bds, jac=True)
            current_theta=res.x
            current_fval = find_opt_weights(get_psi(current_theta,opt_params),opt_params)[0]
            
            if res.success:
                print("Optimization successful after "+str(res.nit)+" Iterations.")
                print(res.message)
            else:
                print("Optimization not successful after "+str(res.nit)+" Iterations.")
                print(res.message)
            print('current theta: ',current_theta)
            print('subproblem value: ',res.fun)
            print('subproblem gradient norm: ', LA.norm(res.jac))
            print('current policy value: ',current_fval)
            print('current norm diff: ', LA.norm(old_theta-current_theta))
        else:
            if np.isinf(opt_params['theta_lb']) and np.isinf(opt_params['theta_ub']):
                res=minimize(CCP_objective, current_theta, args=(opt_params,current_theta,scal_val), method=opt_params['algorithm'], options={'disp': False, 'gtol': 1e-6}, jac=True)
            else:
                res=minimize(CCP_objective, current_theta, args=(opt_params,current_theta,scal_val), method=opt_params['algorithm'], options={'disp': False, 'gtol': 1e-6}, bounds=bds, jac=True)
            current_theta=res.x
            current_fval=find_opt_weights(get_psi(current_theta,opt_params),opt_params)[0]
        
        iteration_counter+=1
    
    if opt_params['verbose']:
        print(" ")
    
    gen_theta=current_theta
    pol_val=find_opt_weights(get_psi(gen_theta,opt_params),opt_params)[0]
        
    return [gen_theta, pol_val]