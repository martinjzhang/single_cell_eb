import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import time
from util import *
import seaborn as sns
from BsplineND import *
from sc_deconv import *


def toy_dist_2d(opt='toy',vis=0):
    if opt=='toy':
        x_A=[0,0.1,0.2,0.3]
        x_B=[0,0.1,0.2,0.3]
        x=[]
        for i in itertools.product(x_A,x_B):
            x.append(i)
        x=np.array(x,dtype=float)
        p=np.zeros([x.shape[0]],dtype=float)

        ## specifying the 2d probability distribution
        p=[0,0.05,0,0,0,0.1,0.1,0.05,0.05,0.2,0.2,0.1,0,0.1,0.05,0]
        p=np.array(p)
    elif opt=='toy_Q':
        alpha = [ 11.9440114,20.80341087,-19.72238357,-3.01209717,-0.08449085,23.06056361,24.95880572,-11.19619854,-1.91011231, -0.11439607, -22.88211576, -10.81795406, -3.52310141, -0.512189,-0.10215229,-3.66013037,-1.93931703,-0.50872035,-0.1965259,-0.10572678, -0.08849988,-0.11567508,-0.10205209,-0.10558915,-0.06873224]
        print(alpha)
        temp1,temp2 = np.meshgrid(np.linspace(0,1,101), np.linspace(0,1,101), indexing='ij')     
        x = np.array([temp1.flatten(),temp2.flatten()]).T    
        Q = Q_gen_ND(x.T,n_degree=5,opt='2d')
        p = px_cal(Q,alpha)
    if vis==1:
        plt.figure()        
        plot_density_2d(p,x, header='mean1=%s, mean2=%s, '%(str(np.sum(p*x[:,0]))[0:4],str(np.sum(p*x[:,1]))[0:4]))
        plt.show()
    return p,x

def data_gen_2d(p,x,N_c=1000,N_r=10,noise='poi',vis=0):
    n_supp=p.shape[0]
    idx_select=np.random.choice(np.arange(n_supp),N_c, p=p,replace=True)
    x_samp = x[idx_select,:]
    Y = np.random.poisson(x_samp*N_r)
    
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
        x_p,x_supp = counts2pdf_2d(x_samp)
        plot_density_2d(x_p,x_supp)
        plt.subplot(122)
        y_p,y_supp = counts2pdf_2d(Y)
        plot_density_2d(y_p,y_supp)
        plt.show()
    
    return x_samp,Y,data_info
        
def counts2pdf_2d(Y):
    if Y.shape[0]==0:
        return np.zeros([2]),np.zeros([2,2])
    Y_supp,idx_reverse=np.unique(Y,return_inverse=True,axis=0)
    Y_pdf=np.bincount(idx_reverse) 
    Y_pdf = Y_pdf / np.sum(Y_pdf)
    return Y_pdf,Y_supp
    

## distribution estimation: density deconvolution
## Y is the count pair n*2 matrix
def dd_2d(Y,noise='poi',gamma=None,c_res=2,c_reg=1e-5,n_degree=5,verbose=False,debug_mode=False):   
    
    ## setting parameters     
    if gamma is None:
        gamma = max(cal_gamma(Y[:,0]),cal_gamma(Y[:,1]))
        
    if verbose: print('n_degree: %s, c_res: %s, c_reg: %s, gamma: %s\n'%(str(n_degree),str(c_res),str(c_reg),str(gamma)))
    
    ## converting the read counts to some sufficient statistics
    idx_low = (Y[:,0]<=gamma)*(Y[:,1]<=gamma) 
    Y_pdf_high,Y_supp_high = counts2pdf_2d(Y[~idx_low,:])    
    Y_pdf,Y_supp = counts2pdf_2d(Y[idx_low,:])   
    n_high = np.sum(~idx_low)
    n_low  = np.sum(idx_low)   
    
    if debug_mode:         
        print('### debug: proportion separation ### start ###')
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plot_density_2d(Y_pdf,Y_supp,header='low proportion: ')
        plt.subplot(122)
        plot_density_2d(Y_pdf_high,Y_supp_high,header='high proportion: ')
        plt.show()
        print('### debug: proportion separation ### end ###\n')

    temp1,temp2 = np.meshgrid(np.linspace(0,1,max(101,int(1/gamma*c_res))), np.linspace(0,1,max(101,int(1/gamma*c_res))), indexing='ij')     
    x       = np.array([temp1.flatten(),temp2.flatten()]).T    
    Q       = Q_gen_ND(x.T,n_degree=n_degree,opt='2d')
    P_model = Pmodel_cal_2d(x,Y_supp,gamma,noise='poi')
    n_param = Q.shape[1]
        
    ## gradient checking
    if debug_mode: 
        print('### debug: proportion separation ### start ###')
        alpha = np.ones([n_param])
        print('Numerical gradients')
        for i in range(5):
            temp = np.zeros([n_param])
            temp[i] += 0.000001
            print((l_cal(alpha+temp,Y_pdf,P_model,Q,c_reg)-l_cal(alpha,Y_pdf,P_model,Q,c_reg))/0.000001)
        print('Close-form gradients')    
        print(grad_cal(alpha,Y_pdf,P_model,Q,c_reg)[0:5]) 
        print('### debug: proportion separation ### end ###\n')
    
    
    ## optimization: using scipy
    res       = sp.optimize.minimize(f_opt,np.zeros([n_param]),args=(Y_pdf,P_model,Q,c_reg),jac=True,options={'disp': False})
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
        plot_density_2d(p_hat,x*gamma,header='low proportion: ')
        #plt.title('')
        plt.subplot(122)
        plot_density_2d(Y_pdf_high,Y_supp_high,header='high proportion: ')
        #plt.title('high proportion')
        plt.show()
        print('### debug: dd result before merging ### end ###\n')
    
    p_hat,x,gamma = p_merge_2d(p_hat,x*gamma,n_low,Y_pdf_high,Y_supp_high-0.11,n_high)
    
    if debug_mode: 
        print('### debug: dd result after merging ### start ###')
        print('gamma:%s'%str(gamma))
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plot_density_2d(p_hat,x*gamma,header='merged result: ')
        plt.show()
        print('### debug: dd result after merging ### end ###\n')
    
    ## record useful information
    dd_info = {}
    dd_info['alpha']    = alpha_hat
    dd_info['x']        = x
    dd_info['MI']       = mutual_info(p_hat,x)
    dd_info['PC']       = pearson_corr(p_hat,x)
    dd_info['gamma']    = gamma
    dd_info['Y_pdf']    = Y_pdf
    dd_info['Y_supp']   = Y_supp
    if verbose:
        plt.figure()
        plot_density_2d(p_hat,x,header='dd_1d: ',range_=[0,1])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show() 
    return p_hat,dd_info

def Pmodel_cal_2d(x,Y_supp,gamma,noise='poi'):
    n_obs  = Y_supp.shape[0]
    n_supp = x.shape[0] 
    P_model_1=sp.stats.poisson.pmf(np.repeat(np.reshape(Y_supp[:,0],[n_obs,1]),n_supp,axis=1),\
                                np.repeat(np.reshape(x[:,0]*gamma,[1,n_supp]),n_obs,axis=0))
    P_model_2=sp.stats.poisson.pmf(np.repeat(np.reshape(Y_supp[:,1],[n_obs,1]),n_supp,axis=1),\
                                np.repeat(np.reshape(x[:,1]*gamma,[1,n_supp]),n_obs,axis=0))
    P_model=P_model_1*P_model_2
    return P_model

def p_merge_2d(p1,x1,n1,p2,x2,n2): 
    ## only care about non-zero parts 
    x_c=np.max(x1,axis=0)
    x_step = x1[1,1]-x1[0,1]
    x1 = x1[p1>0,:]
    p1 = p1[p1>0]
    x2 = x2[p2>0,:]
    p2 = p2[p2>0]  
        
    ## matching the support of the second set 
    for i in range(x2.shape[0]):
        if x2[i,0]<=x_c[0]:
            x2[i,0] = x1[np.argmin(np.absolute(x1[:,0]-x2[i,0])),0]
        if x2[i,1]<=x_c[1]:
            x2[i,1] = x1[np.argmin(np.absolute(x1[:,1]-x2[i,1])),1]
    
    #print('debugging')
    #for i in range(x2.shape[0]):
    #    if (x2[i,0]<=x_c[0]) and (x2[i,0] not in x1[:,0]):
    #        print(x2[i,:])
    #    if (x2[i,1]<=x_c[1]) and (x2[i,1] not in x1[:,1]):
    #        print(x2[i,:])
    
            
    ## combining the pdf   
    p1        = p1*n1/(n1+n2)
    p2        = p2*n2/(n1+n2)
    p_all     = np.concatenate([p1,p2])
    x_all     = np.concatenate([x1,x2])
    gamma     = np.max(x_all)
    x_all    /= gamma
    return p_all,x_all,gamma    
    
def ml_2d(Y): # with no rounding to the nearest neighbour
    Y_pdf,Y_supp = counts2pdf_2d(Y)
    N_c          = Y.shape[0]
    gamma        = max(cal_gamma(Y[:,0]),cal_gamma(Y[:,1]))
    p_hat        = Y_pdf
    x            = Y_supp/gamma
    
    # recording the information
    ml_info={}
    ml_info['Y_pdf']  = Y_pdf
    ml_info['Y_supp'] = Y_supp
    ml_info['x']      = x
    ml_info['N_c']    = N_c
    ml_info['gamma']  = gamma
    return p_hat,ml_info

## visualization of the distribution estimation result
def plot_result_2d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info):
    ## load some important parameters
    x    = data_info['x'] # the true support 
    x_dd = dd_info['x']*dd_info['gamma']/data_info['N_r'] # the support assumed by deconv
    x_ml = ml_info['x']*ml_info['gamma']/data_info['N_r']
    
    true_pc = pearson_corr(p,data_info['x'])
    true_mi = mutual_info(p,data_info['x'])
    dd_pc   = pearson_corr(p_hat,dd_info['x'])
    dd_mi   = mutual_info(p_hat,dd_info['x'])
    ml_pc   = pearson_corr(p_hat_ml,ml_info['x'])
    ml_mi   = mutual_info(p_hat_ml,ml_info['x'])
    
    plt.figure(figsize=[18,5])
    plt.subplot(1,3,1)
    plot_density_2d(p,x,header='True: ',range_=[0,1])
    plt.subplot(1,3,2)
    plot_density_2d(p_hat,x_dd,header='dd_2d: ',range_=[0,1])
    plt.subplot(1,3,3)
    plot_density_2d(p_hat_ml,x_ml,header='ml_2d: ',range_=[0,1])
    plt.show()
    
    
## moments estimation
def dd_moments_2d(Y,k=2,gamma=None,noise='poi'):
    ## converting the read counts to some sufficient statistics
    Y_pdf,Y_supp = counts2pdf_2d(Y)
    ## parameter setting   
    N_c    = Y.shape[0]
    if gamma is None: gamma  = max(cal_gamma(Y[:,0]),cal_gamma(Y[:,1]))
    
    M_hat = np.zeros([k+1,k+1],dtype=float)   
    for k1 in range(k+1):
        for k2 in range(k+1):
            for j in range(Y_supp.shape[0]):
                temp = 1
                for l in range(k1):
                    temp *= Y_supp[j,0]-l
                for l in range(k2):
                    temp *= Y_supp[j,1]-l
                M_hat[k1,k2] += Y_pdf[j] * temp
            M_hat[k1,k2] /= gamma**(k1+k2)
    
    mean1 = M_hat[1,0]
    mean2 = M_hat[0,1]
    var1  = M_hat[2,0]-M_hat[1,0]**2
    var2  = M_hat[0,2]-M_hat[0,1]**2
    cov12 = M_hat[1,1]-M_hat[1,0]*M_hat[0,1]
    PC    = cov12/np.sqrt(var1*var2)
    return M_hat,mean1,mean2,var1,var2,cov12,PC,gamma

def M_convert_2d(M,N_r1,N_r2):
    M2 = np.zeros(M.shape,dtype=float)
    r  = N_r1/N_r2
    for k1 in range(M2.shape[0]):
        for k2 in range(M2.shape[0]):
            M2[k1,k2] = M[k1,k2]*r**(k1+k2)
    mean1 = M2[1,0]
    mean2 = M2[0,1]
    var1  = M2[2,0]-M2[1,0]**2
    var2  = M2[0,2]-M2[0,1]**2
    cov12 = M2[1,1]-M2[1,0]*M2[0,1]
    PC    = cov12/np.sqrt(var1*var2)
    return M2,mean1,mean2,var1,var2,cov12,PC