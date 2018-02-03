import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import time
from util import *
import seaborn as sns

## toy distribution
def toy_dist(opt='1d',vis=0):
    if opt == '1d_Q':
        alpha = np.array([2.4,2.15,1.2,-0.7 ,-5])
        x     = np.linspace(0,1,101)
        Q     = Q_gen()
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if vis ==1:
            print(alpha)
    elif opt == '1d_CD3E':
        alpha = np.array([2.29,4.04,3.6,-2.05 ,-7.89])
        x     = np.linspace(0,1,101)
        Q     = Q_gen()
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if vis ==1:
            print(alpha)
    elif opt == '1d_FTL':
        alpha = np.array([ 0.87521866, -0.43979061, -0.33585905,  2.21903186, -0.81100455, -1.62846978, -1.23172219])
        x     = np.linspace(0,1,101)
        Q     = Q_gen(n_degree=7)
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if vis ==1:
            print(alpha)
            
    elif opt == '1d': 
        p = np.array([0,0.2,0.3,0.4,0.1,0,0,0,0,0])
        x = np.array([0,0.05,0.15,0.25,0.35,0.4,0.5,0.6,0.7,1.0])
    else: 
        print('Err: option not recognized!')
        
    if vis == 1:
        plt.figure()
        plot_density_1d(p,x)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('toy distribution')
        plt.legend()
        plt.show()   
    return p,x

## data generation
def data_gen_1d(p,x,N_c,N_r,noise='poi',vis=0):
    n_supp = p.shape[0]
    x_samp = np.random.choice(x,N_c, p=p,replace=True)
    if noise == 'poi':
        Y=np.random.poisson(x_samp*N_r)
    elif noise == 'bin':
        Y=np.random.binomial(N_r,x_samp)
    ## recording the data information
    data_info={}
    data_info['x']     = x
    data_info['p']     = p
    data_info['N_c']   = N_c
    data_info['N_r']   = N_r
    data_info['noise'] = noise
    
    if vis==1:
        plt.figure(figsize=[12,5])
        plt.subplot(121)
        plt.hist(x_samp,bins=10)
        plt.subplot(122)
        #plt.hist(Y,bins=np.arange(15)-0.5)
        sns.distplot(Y)
        plt.show()
    return x_samp,Y,data_info


## distribution estimation: density deconvolution
def dd_1d(Y,noise='poi',verbose=False,gamma=None,resolution=2):   
    n_degree = 7
     
    
    ## setting parameters 
    if gamma is None:
        gamma = cal_gamma(Y)
        if verbose: print('gamma:%s'%str(gamma))
    
    ## converting the read counts to some sufficient statistics
    Y_pdf_high,Y_supp_high = counts2pdf_1d(Y[Y>=gamma])    
    Y_pdf,Y_supp = counts2pdf_1d(Y[Y<gamma])   
    n_high = np.sum(Y>=gamma)
    n_low  = np.sum(Y<gamma)
            
    x       = np.linspace(0,1,max(101,int(1/gamma*resolution)))
    Q       = Q_gen(x,n_degree=n_degree)
    P_model = Pmodel_cal(x,Y_supp,gamma,noise='poi')
    
    ## gradient checking
    #alpha = np.ones([n_degree])
    #for i in range(5):
    #    temp = np.zeros([5])
    #    temp[i] += 0.000001
    #    print((l_cal(alpha+temp,Y_pdf,P_model,Q)-l_cal(alpha,Y_pdf,P_model,Q))/0.000001)
    #    
    #print(grad_cal(alpha,Y_pdf,P_model,Q)) 
    
    ## optimization: using scipy
    res       = sp.optimize.minimize(f_opt,np.zeros([n_degree]),args=(Y_pdf,P_model,Q),jac=True,options={'disp': False})
    alpha_hat = res.x
    print('alpha_hat: ', alpha_hat)
    _         = l_cal(alpha_hat,Y_pdf,P_model,Q,verbose=True)
    p_hat     = px_cal(Q,alpha_hat)
    
    p_hat,x,gamma = p_merge(p_hat,x*gamma,n_low, Y_pdf_high,Y_supp_high,n_high)
    
    ## record useful information
    dd_info = {}
    dd_info['alpha']    = alpha_hat
    dd_info['x']        = x
    dd_info['gamma']    = gamma
    dd_info['Y_pdf']    = Y_pdf
    dd_info['Y_supp']   = Y_supp
    if verbose:
        plt.figure()
        plot_density_1d(p_hat,x)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('dd_1d')
        plt.legend()
        plt.show() 
    return p_hat,dd_info

def f_opt(alpha,Y_pdf,P_model,Q):
    return -l_cal(alpha,Y_pdf,P_model,Q),-grad_cal(alpha,Y_pdf,P_model,Q)

def px_cal(Q,alpha):
    P_X  = np.exp(Q.dot(alpha))
    P_X /= np.sum(P_X)
    return P_X
    
def l_cal(alpha,Y_pdf,P_model,Q,verbose=False):
    P_X = px_cal(Q,alpha)
    #l   = np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-8))  
    #print(np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-8)),0.01*np.sum(alpha**2))
    l   = np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-8)) - 1e-4*np.sum(alpha**2)
    if verbose is True: print('-l:%s, reg:%s'%(str(-np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-8))),str(1e-4*np.sum(alpha**2))))
    return l
    

def grad_cal(alpha,Y_pdf,P_model,Q):    
    P_X = px_cal(Q,alpha) # P_X        
    P_Y = P_model.dot(P_X) # P_Y    
    W = (((P_model.T/(P_Y+1e-8)).T-1)*P_X).T # gradient
    #grad = Q.T.dot(W.dot(Y_pdf))
    grad = Q.T.dot(W.dot(Y_pdf)) - 1e-4*2*alpha
    return grad
    
def Q_gen(x=None,vis=0,n_degree=5): # generating a natural spline from B-spline basis
    #n_degree=5
    print('n_degree:%s'%str(n_degree))
    if x is None: x = np.linspace(0,1,101)
    t = np.linspace(0,1,3+n_degree+1)
    Q = np.zeros([x.shape[0],n_degree],dtype=float)
    for i in range(n_degree):
        c = np.zeros([n_degree])
        c[i]=1
        spl=sp.interpolate.BSpline(t=t,c=c,k=3)
        Q[:,i] = spl(x)
    if vis == 1:
        plt.figure(figsize=[16,5])
        for i in range(n_degree):
            plt.subplot(121)
            plt.plot(x,Q[:,i],label=str(i+1))
            plt.subplot(122)
            plt.plot(x,Q[:,i],label=str(i+1))
            plt.ylim([-2,2])
        plt.legend()
        plt.show()
    return Q

def p_merge(p1,x1,n1,p2,x2,n2):    
    ## combining the pdf
    p1       = p1*n1/(n1+n2)
    p2       = p2*n2/(n1+n2)
    p_all    = np.concatenate([p1,p2])
    x_all    = np.concatenate([x1,x2])
    sort_idx = np.argsort(x_all)
    
    p = [p_all[sort_idx[0]]]
    x = [x_all[sort_idx[0]]]
    for i in range(1,sort_idx.shape[0]):
        if x_all[sort_idx[i]] == x[-1]:
            p[-1] += p_all[sort_idx[i]]
        else:
            p.append(p_all[sort_idx[i]])
            x.append(x_all[sort_idx[i]])
    
    ## transforming to a new set of support   
    x_step     = (np.max(x1)-np.min(x1))/(len(x1)-1)
    x_new      = np.linspace(np.min(x),np.max(x),(np.max(x)-np.min(x))/x_step+1)
    cdf        = np.cumsum(p)
    p_new      = np.interp(x_new,x,cdf)
    p_new[1:] -= p_new[0:-1]
    gamma = np.max(x_new) 
    x_new /= gamma
   
    return p_new,x_new,gamma
    
#def Q_gen(x=None,vis=0): # generating a natural spline from B-spline basis
#    if x is None: x = np.linspace(0,1,101)
#    t = np.linspace(0,1,9)
#    Q = np.zeros([x.shape[0],5],dtype=float)
#    for i in range(5):
#        c = np.zeros([5])
#        c[i]=1
#        spl=sp.interpolate.BSpline(t=t,c=c,k=3)
#        Q[:,i] = spl(x)
#    if vis == 1:
#        plt.figure()
#        for i in range(5):
#            plt.plot(x,Q[:,i],label=str(i+1))
#        plt.legend()
#        plt.show()
#    return Q

def counts2pdf_1d(Y):
    Y_pdf=np.bincount(Y)
    Y_pdf=Y_pdf/Y.shape[0]
    Y_supp=np.arange(Y_pdf.shape[0])
    return Y_pdf,Y_supp

def cal_gamma(Y): # we use this function to define the data driven gamma for a
    Y_99  = np.percentile(Y,99)
    gamma = int(Y_99+3*np.sqrt(Y_99)) # gamma should be roughly Y_max. The 95% percentile is used for robustness consideration
    return gamma

def Pmodel_cal(x,Y_supp,N_r,noise='poi'):
    n_supp = x.shape[0]
    n_obs  = Y_supp.shape[0]
    if noise == 'poi':
        P_model=sp.stats.poisson.pmf(np.repeat(np.reshape(np.arange(n_obs),[n_obs,1]),n_supp,axis=1),\
                                    np.repeat(np.reshape(x*N_r,[1,n_supp]),n_obs,axis=0))
    elif noise == 'bin':
        P_model=sp.stats.binom.pmf(np.repeat(np.reshape(np.arange(n_obs),[n_obs,1]),n_supp,axis=1),N_r,\
                                    np.repeat(np.reshape(x,[1,n_supp]),n_obs,axis=0))
    else: 
        print('Err: noise type not recognized!')
        return
    return P_model

## maximum likelihood estimation
def ml_1d(Y): # with no rounding to the nearest neighbour
    Y_pdf,Y_supp = counts2pdf_1d(Y)
    N_c          = Y.shape[0]
    N_r          = cal_gamma(Y)
    p_hat        = Y_pdf
    x            = Y_supp/(N_r+0.0)
    
    # recording the information
    ml_info={}
    ml_info['Y_pdf']  = Y_pdf
    ml_info['Y_supp'] = Y_supp
    ml_info['x']      = x
    ml_info['N_r']    = N_r
    return p_hat,ml_info

## moments estimation
def dd_moments_1d(Y,k=2,noise='poi'):
    ## converting the read counts to some sufficient statistics
    Y_pdf,Y_supp = counts2pdf_1d(Y)
    ## parameter setting   
    N_c    = Y.shape[0]
    gamma  = cal_gamma(Y)
    
    M_hat = np.zeros(k)   
    if noise == 'poi':
        for i in range(k):
            for j in range(Y_supp.shape[0]):
                temp = 1
                for l in range(i+1):
                    temp *= Y_supp[j]-l
                M_hat[i] += Y_pdf[j] * temp
            M_hat[i] /= (gamma**(i+1)+0.0)
    if noise == 'bin':
        for i in range(k):
            for j in range(Y_supp.shape[0]):
                if Y_supp[j] >= i+1:
                    M_hat[i] += Y_pdf[j]*sp.special.comb(Y_supp[j], i+1)/(sp.special.comb(gamma, i+1)+0.0)
    mean_hat = M_hat[0]
    var_hat  = M_hat[1]-M_hat[0]**2
    return mean_hat,var_hat,M_hat,gamma

def M_convert(M,N_r1,N_r2):
    M2 = np.zeros(M.shape,dtype=float)
    r  = N_r1/(N_r2+0.0)
    for i in range(M2.shape[0]):
        M2[i] = M[i]*r**(i+1)
    mean2 = M2[0]
    var2  = M2[1]-M2[0]**2
    return mean2,var2,M2

## visualization of the distribution estimation result
def plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info):
    ## load some important parameters
    x    = data_info['x'] # the true support 
    x_dd = dd_info['x']*dd_info['gamma']/(data_info['N_r']+0.0) # the support assumed by deconv
    x_ml = ml_info['x']*dd_info['gamma']/(data_info['N_r']+0.0)
    
    ## ml estimation
    err_dd = dist_W1(p,p_hat,x,x_dd)
    err_ml = dist_W1(p,p_hat_ml,x,x_ml)
    
    if len(dd_info['Y_supp'].shape)==1:
        # first figure: the confidence interval
        P_model = Pmodel_cal(x,dd_info['Y_supp'],data_info['N_r'],noise=data_info['noise'])
        x_s = np.linspace(0,1,101)
        plt.figure(figsize=[18,5])
        plt.subplot(121)
        plt.plot(x_s,supp_trans(p_hat_ml,x_ml,x_s),linewidth=4,label='ml: %s'%str(err_ml)[0:6],alpha=0.6)
        plt.plot(x_s,supp_trans(p_hat,x_dd,x_s),linewidth=4,label='deconv: %s'%str(err_dd)[0:6],alpha=0.6)
        plt.plot(x_s,supp_trans(p,x,x_s),linewidth=4,label='true distribution',alpha=0.6)
        plt.legend()
        plt.title('pdf')
        plt.subplot(122)
        plt.plot(x_ml,np.cumsum(p_hat_ml),marker='o',label='ml: %s'%str(err_ml)[0:6],alpha=0.6)
        plt.plot(x_dd,np.cumsum(p_hat),marker='o',label='deconv: %s'%str(err_dd)[0:6],alpha=0.6)
        plt.plot(x,np.cumsum(p),marker='o',label='true distribution',alpha=0.6)
        plt.title('cdf')
        plt.legend()
        plt.show()
    return err_dd,err_ml

def supp_trans(p,x,x_new):
    cdf        = np.cumsum(p)
    p_new      = np.interp(x_new,x,cdf)
    p_new[1:] -= p_new[0:-1]
    return p_new