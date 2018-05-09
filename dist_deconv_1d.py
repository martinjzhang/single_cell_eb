import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import time
from util import *
import seaborn as sns
from BsplineND import *
from multiprocessing import Pool
import time

## distribution estimation: density deconvolution
def dd_1d(Y,noise='poi',gamma=None,c_res=2,c_reg=1e-5,n_degree=5,zero_inflate=False,verbose=False,debug_mode=False):   
    
    ## setting parameters     
    if gamma is None:
        gamma = cal_gamma(Y)
        
    if verbose: print('n_degree: %s, c_res: %s, c_reg: %s, gamma: %s\n'%(str(n_degree),str(c_res),str(c_reg),str(gamma)))
        
    
    ## converting the read counts to some sufficient statistics
    #Y_high,n_high = Y[Y>1000],np.sum(Y>1000)
    #Y_low,n_low   = Y[Y<=1000],np.sum(Y<=1000)
    #Y_pdf_high,Y_supp_high = counts2pdf_1d(Y_high)    
    #Y_pdf,Y_supp = counts2pdf_1d(Y_low)   
    
    gamma_ = 20*(gamma<20)+int(gamma+np.sqrt(gamma))*(gamma>=20)
    Y_pdf_high,Y_supp_high = counts2pdf_1d(Y[Y>gamma_])    
    Y_pdf,Y_supp = counts2pdf_1d(Y[Y<=gamma_])   
    n_high = np.sum(Y>gamma_)
    n_low  = np.sum(Y<=gamma_)   
    
    #Y_pdf_high,Y_supp_high = counts2pdf_1d(Y[Y>100])    
    #Y_pdf,Y_supp = counts2pdf_1d(Y)   
    #n_high = np.sum(Y>100)
    #n_low  = np.sum(Y<100) 
    
    if debug_mode: 
        print('### debug: proportion separation ### start ###')
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plt.plot(Y_supp,Y_pdf*n_low,marker='o')
        plt.title('low proportion')
        plt.subplot(122)
        plt.plot(Y_supp_high,Y_pdf_high*n_high,marker='o')
        plt.title('high proportion')
        plt.show()
        print('### debug: proportion separation ### end ###\n')
        
    x          = np.linspace(0,1,max(101,int(1/gamma*c_res)))
    Q,n_degree = Q_gen_ND(x,n_degree=n_degree,zero_inflate=zero_inflate)
        
    P_model = Pmodel_cal(x,Y_supp,gamma,noise='poi')
    
    ## gradient checking
    if debug_mode: 
        print('### debug: proportion separation ### start ###')
        alpha = np.ones([n_degree])
        print('Numerical gradients')
        for i in range(n_degree):
            temp = np.zeros([n_degree])
            temp[i] += 0.000001
            print((l_cal(alpha+temp,Y_pdf,P_model,Q,c_reg)-l_cal(alpha,Y_pdf,P_model,Q,c_reg))/0.000001)
        print('Close-form gradients')    
        print(grad_cal(alpha,Y_pdf,P_model,Q,c_reg)) 
        print('### debug: proportion separation ### end ###\n')
    
    
    ## optimization: using scipy
    res       = sp.optimize.minimize(f_opt,np.zeros([n_degree]),args=(Y_pdf,P_model,Q,c_reg),jac=True,options={'disp': False})
    alpha_hat = res.x
    
    if debug_mode: 
        print('### debug: optimization ### start ###')
        print('c_reg',c_reg)
        _         = l_cal(alpha_hat,Y_pdf,P_model,Q,c_reg,verbose=True)
        print('### debug: optimization ### end ###\n')
    p_hat     = px_cal(Q,alpha_hat)
    
    if debug_mode: 
        print('### debug: dd result before merging ### start ###')
        print('alpha_hat: ', alpha_hat)
        print('gamma:%s'%str(gamma))
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plt.plot(x*gamma,p_hat*n_low,marker='o')
        plt.title('low proportion')
        plt.subplot(122)
        plt.plot(Y_supp_high,Y_pdf_high*n_high,marker='o')
        plt.title('high proportion')
        plt.show()
        print('### debug: dd result before merging ### end ###\n')
    
    p_hat,x,gamma = p_merge(p_hat,x*gamma,n_low, Y_pdf_high,Y_supp_high,n_high)
    
    if debug_mode: 
        print('### debug: dd result after merging ### start ###')
        print('gamma:%s'%str(gamma))
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plt.plot(x*gamma,p_hat*n_low,marker='o')
        plt.title('merged result')
        plt.show()
        print('### debug: dd result after merging ### end ###\n')
    
    ## record useful information
    dd_info = {}
    dd_info['p']        = p_hat
    dd_info['alpha']    = alpha_hat
    dd_info['x']        = x
    dd_info['gamma']    = gamma
    dd_info['entropy']  = entropy(p_hat,x)
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
    return p_hat,x

def f_opt(alpha,Y_pdf,P_model,Q,c_reg):
    return -l_cal(alpha,Y_pdf,P_model,Q,c_reg),-grad_cal(alpha,Y_pdf,P_model,Q,c_reg)

def px_cal(Q,alpha):
    P_X  = np.exp(Q.dot(alpha))
    P_X /= np.sum(P_X)
    return P_X
    
def l_cal(alpha,Y_pdf,P_model,Q,c_reg,verbose=False):
    P_X = px_cal(Q,alpha)
    l   = np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-10)) - c_reg*np.sum(alpha**2)
    if verbose is True: print('-l:%s, reg:%s'%(str(-np.sum(Y_pdf*np.log(P_model.dot(P_X)+1e-8))),str(c_reg*np.sum(alpha**2))))
    return l
    

def grad_cal(alpha,Y_pdf,P_model,Q,c_reg):    
    P_X = px_cal(Q,alpha) # P_X        
    P_Y = P_model.dot(P_X) # P_Y    
    W = (((P_model.T/(P_Y+1e-10)).T-1)*P_X).T # gradient
    #grad = Q.T.dot(W.dot(Y_pdf))
    grad = Q.T.dot(W.dot(Y_pdf)) - c_reg*2*alpha
    return grad
    
def Q_gen(x=None,vis=0,n_degree=5,zero_inflate=False): # generating a natural spline from B-spline basis
    #n_degree=5
    # print('n_degree:%s'%str(n_degree))
    if x is None: x = np.linspace(0,1,101)
    t = np.linspace(0,1,3+n_degree+1)
    Q = np.zeros([x.shape[0],n_degree],dtype=float)
    for i in range(n_degree):
        c = np.zeros([n_degree])
        c[i]=1
        spl=sp.interpolate.BSpline(t=t,c=c,k=3)
        Q[:,i] = spl(x)
    if zero_inflate:
        Q_t        = np.zeros([Q.shape+1])
        Q_t[0,0]   = 1
        Q_t[1:,1:] = Q
        Q = Q_t
        n_degree += 1
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
    return Q,n_degree

def p_merge(p1,x1,n1,p2,x2,n2):    
    ## only care about non-zero parts 
    x_c=x1[-1]
    x_step = x1[1]-x1[0]
    x1 = x1[p1>0]
    p1 = p1[p1>0]
    x2 = x2[p2>0]
    p2 = p2[p2>0]    
        
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
    x_new      = np.linspace(np.min(x),np.max(x),(np.max(x)-np.min(x))/x_step+1)
    cdf        = np.cumsum(p)
    p_new      = np.interp(x_new,x,cdf)
    p_new[1:] -= p_new[0:-1]
    gamma = np.max(x_new) 
    
    ## smooth the connection region
    #x_idx = (x_new>(x_c-2*np.sqrt(x_c))) * (x_new<(x_c+2*np.sqrt(x_c)))    
    #for i in range(x_new.shape[0]):
    #    if x_new[i]>x_c-2*np.sqrt(x_c):# and x_new[i]<x_c+x_width:
    #        temp_alpha = min(1,(x_new[i] - (x_c-2*np.sqrt(x_c)) ) / np.sqrt(x_c))
    #        x_width = 2*np.sqrt(x_new[i])    
    #        temp_idx = (x_new>(x_new[i]-x_width)) * (x_new<(x_new[i]+x_width))
    #        temp_x = x_new[temp_idx]
    #        temp_sigma = x_width/3
    #        kernel_gaussian = 1/(np.sqrt(2*np.pi*temp_sigma**2))*np.exp(-(temp_x-x_new[i])**2/2/temp_sigma**2)
    #        kernel_gaussian /= np.sum(kernel_gaussian)
    #        #print(kernel_gaussian)
    #        #break
    #        
    #        temp_p = p_new[i]*temp_alpha
    #        p_new[i] *=(1-temp_alpha)
    #        p_new[temp_idx] += temp_p*kernel_gaussian
    
    x_new /= gamma
    
    return p_new,x_new,gamma

def counts2pdf_1d(Y):
    Y_pdf=np.bincount(Y)
    Y_pdf=Y_pdf/Y.shape[0]
    Y_supp=np.arange(Y_pdf.shape[0])
    return Y_pdf,Y_supp

def cal_gamma(Y): # we use this function to define the data driven gamma 
    if Y.max() < 50:
        return int(Y.max()+np.sqrt(Y.max()))
    
    Y_ct  = np.bincount(Y)
    Y_ct  = np.append(Y_ct,[0])
    Y_99  = np.percentile(Y,90)
    temp  = np.arange(Y_ct.shape[0])
    gamma = np.where((Y_ct<5)&(temp>Y_99))[0][0]
    #print(Y_99)
    #print(np.where(Y_ct<10))
    return min(gamma,150)


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