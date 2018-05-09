from scipy.stats import beta
import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import time
from util import *
import seaborn as sns
from BsplineND import *
from sc_deconv import *
from multiprocessing import Pool
import time

def Px_gen_beta(verbose=False):
    theta = np.array([1,20,8,20,0.5])
    a1,b1,a2,b2,p = theta
    x = np.linspace(1e-4,1-1e-4,101)
    p = p_cal_beta(theta,x)
    print('theta',theta)
    print(M_cal_beta(theta,x,5))
    if verbose:
        plt.figure()
        plot_density_1d(p,x)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('toy distribution, mean: %s'%str(np.sum(p*x)))
        plt.legend()
        plt.show()     
    return p,x

def dd_1d_beta(Y,N_r,rep_time=5,verbose=False,K=5):
    Y_pdf,Y_supp = counts2pdf_1d(Y)   
    _,_,M,N_r_hat = dd_moments_1d(Y,k=K,noise='poi')
    _,_,M = M_convert(M,N_r_hat,N_r)    
    x = np.linspace(1e-4,1-1e-4,101)
    
    if verbose: 
        print('estimated moments',M)
    ## optimization
    cons = ({'type': 'ineq', 'fun': lambda x:  x},{'type': 'ineq', 'fun': lambda x: 1-x[4]})
    #best_res=None
    #for i in range(rep_time):
    theta_0 = np.array([1,20,5,20,0.5])    
    res = sp.optimize.minimize(f_opt_beta,theta_0,args=(x,M),jac=False,constraints=cons,options={'disp': False})
    #if best_res is None: best_res = res
    #if best_res.fun>res.fun is None: best_res = res 
    a1,b1,a2,b2,p = res.x
    if verbose: 
        print('theta',res.x)
    p_hat = p_cal_beta(res.x,x)
    
    if verbose:
        print('reconstructed moments',M_cal_px(p_hat,x,K))
    ## record useful information
    dd_info = {}
    dd_info['p']        = p_hat
    dd_info['theta']    = res.x
    dd_info['x']        = x
    dd_info['gamma']    = N_r
    dd_info['entropy']  = entropy(p_hat,x)
    dd_info['Y_pdf']    = Y_pdf
    dd_info['Y_supp']   = Y_supp
    return p_hat,dd_info

def f_opt_beta(theta,x,M):
    return f_cal_beta(theta,x,M)

def f_cal_beta(theta,x,M):
    M_hat=M_cal_beta(theta,x,M.shape[0])
    return np.sum(((M_hat-M)/np.sqrt(M))**2)

def p_cal_beta(theta,x):
    a1,b1,a2,b2,p = theta
    p1 = beta.pdf(x,a1,b1)
    p2 = beta.pdf(x,a2,b2)      
    p1 = p1/np.sum(p1)
    p2 = p2/np.sum(p2)
    return p*p1+(1-p)*p2

def M_cal_beta(theta,x,K):
    a1,b1,a2,b2,p = theta
    p1 = beta.pdf(x,a1,b1)
    p2 = beta.pdf(x,a2,b2)      
    p1 = p1/np.sum(p1)
    p2 = p2/np.sum(p2)
    M = np.zeros([K],dtype=float)
    for k in range(K):
        M[k] = p*p1.dot(x**(k+1)) + (1-p)*p2.dot(x**(k+1))
    return M

def M_cal_px(p,x,K):
    M = np.zeros([K],dtype=float)
    for k in range(K):
        M[k] = p.dot(x**(k+1))
    return M

def f_grad_beta(tehta,x,M):
    pass

def fit_beta_mixture(Y,gamma,n_itr=100):
    Y = Y/gamma
    Y = Y[(Y>0) & (Y<1)]
    #plt.figure(figsize=[12,5])
    #sns.distplot(Y)
    #plt.show()
    
    theta = np.array([1,10,2,5,0.5])
    w     = np.zeros([Y.shape[0],2],dtype=float)
    
    if_converge = False
    for i in range(n_itr):
        theta_old = theta
        # E step
        w[:,0] = beta.pdf(Y,theta[0],theta[1])*theta[4]
        w[:,1] = beta.pdf(Y,theta[2],theta[3])*(1-theta[4])
        w      = (w.T/w.sum(axis=1)).T
           
        # M step
        rand_idx = np.random.binomial(1,w[:,0],size=Y.shape[0])
        theta[4] = w[:,0].mean()
        print(np.sum(rand_idx==1))
        theta[0],theta[1],_,_ = beta.fit(Y[rand_idx==1],loc=0,scale=1)
        theta[2],theta[3],_,_ = beta.fit(Y[rand_idx==0],loc=0,scale=1)
        
        
        
        
        if np.linalg.norm(theta-theta_old) < 1e-8:
            if_converge = True
            break
    
    return theta,if_converge

def beta_fit():
    pass