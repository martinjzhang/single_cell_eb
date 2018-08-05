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

"""
    toy distribution
"""
def toy_dist(opt='1d',verbose=False,Y=None):
    np.random.seed(42)
    if opt == '1d_Q':
        alpha = np.array([2.4,2.15,1.2,-0.7 ,-5])
        x     = np.linspace(0,1,101)
        Q,_   = Q_gen()
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if verbose:
            print(alpha)
    elif opt == '1d_CD3E':
        alpha = np.array([2.29,4.04,3.6,-2.05 ,-7.89])
        x     = np.linspace(0,1,101)
        Q,_   = Q_gen()
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if verbose:
            print(alpha)
    elif opt == '1d_FTL':
        alpha = np.array([ 0.87521866, -0.43979061, -0.33585905,  2.21903186, -0.81100455, -1.62846978, -1.23172219])
        x     = np.linspace(0,1,101)
        Q     = Q_gen(n_degree=7)
        p     = np.exp(Q.dot(alpha))
        p    /= np.sum(p)
        if verbose:
            print(alpha)
            
    elif opt == '1d': 
        p = np.array([0,0.2,0.3,0.4,0.1,0,0,0,0,0])
        x = np.array([0,0.05,0.15,0.25,0.35,0.4,0.5,0.6,0.7,1.0])
        
    elif opt == 'export':
        p,x = counts2pdf_1d(Y)
        x   = x/np.percentile(Y,99)*0.75
        p   = p[x<1]
        x   = x[x<1]
        p   = p/p.sum()
        
    else: 
        print('Err: option not recognized!')
        
    if verbose:
        plt.figure()
        plot_density_1d(p,x)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('toy distribution, mean: %s'%str(np.sum(p*x)))
        plt.legend()
        plt.show()   
    return p,x

"""
    data generation
"""
def data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=False):
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
    
    if verbose:
        plt.figure(figsize=[16,5])
        plt.subplot(121)
        plt.hist(x_samp,bins=10)
        plt.subplot(122)
        #plt.hist(Y,bins=np.arange(15)-0.5)
        plt.title('mean_count=%0.2f'%Y.mean())
        sns.distplot(Y)
        plt.show()
    return x_samp,Y,data_info


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
    return p_hat,dd_info

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
    x_idx = (x_new>(x_c-2*np.sqrt(x_c))) * (x_new<(x_c+2*np.sqrt(x_c)))    
    for i in range(x_new.shape[0]):
        if x_new[i]>x_c-2*np.sqrt(x_c):# and x_new[i]<x_c+x_width:
            temp_alpha = min(1,(x_new[i] - (x_c-2*np.sqrt(x_c)) ) / np.sqrt(x_c))
            x_width = 2*np.sqrt(x_new[i])    
            temp_idx = (x_new>(x_new[i]-x_width)) * (x_new<(x_new[i]+x_width))
            temp_x = x_new[temp_idx]
            temp_sigma = x_width/3
            kernel_gaussian = 1/(np.sqrt(2*np.pi*temp_sigma**2))*np.exp(-(temp_x-x_new[i])**2/2/temp_sigma**2)
            kernel_gaussian /= np.sum(kernel_gaussian)
            #print(kernel_gaussian)
            #break
            
            temp_p = p_new[i]*temp_alpha
            p_new[i] *=(1-temp_alpha)
            p_new[temp_idx] += temp_p*kernel_gaussian
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
    
    
    
    #Y_ct  = np.bincount(Y)
    #gamma = np.where(Y_ct<5)[0]
    #
    #Y_99  = np.percentile(Y,99)
    #gamma = int(Y_99+3*np.sqrt(Y_99)) # gamma should be roughly Y_max. The 95% percentile is used for robustness consideration
    #
    #return min(gamma,100)
    #return min(gamma,120)

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
    gamma        = cal_gamma(Y)
    p_hat        = Y_pdf
    x            = Y_supp/gamma
    
    # recording the information
    ml_info={}
    ml_info['Y_pdf']  = Y_pdf
    ml_info['Y_supp'] = Y_supp
    ml_info['x']      = x
    ml_info['gamma']  = gamma
    return p_hat,ml_info

## moments estimation
def dd_moments_1d(Y,k=2,noise='poi',gamma=None,size_factor=None):
    ## cell specific normalization factor 
    if size_factor is None: 
        size_factor = 1
    else:
        size_factor = size_factor.clip(min=0.1)
    
    ## converting the read counts to some sufficient statistics
    Y_pdf,Y_supp = counts2pdf_1d(np.array(Y/size_factor,dtype=int))
    ## parameter setting   
    N_c    = Y.shape[0]
    if gamma is None: gamma = cal_gamma(Y)
    
    M_hat = np.zeros(k)   
    if noise == 'poi':
        for i in range(k):
            for j in range(Y_supp.shape[0]):
                temp = 1
                for l in range(i+1):
                    temp *= Y_supp[j]-l
                M_hat[i] += Y_pdf[j] * temp
            M_hat[i] /= gamma**(i+1)
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
    r  = N_r1/N_r2
    for i in range(M2.shape[0]):
        M2[i] = M[i]*r**(i+1)
    mean2 = M2[0]
    var2  = M2[1]-M2[0]**2
    return mean2,var2,M2

## visualization of the distribution estimation result
def plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info,verbose=False):
    ## load some important parameters
    x    = data_info['x'] # the true support 
    x_dd = dd_info['x']*dd_info['gamma']/(data_info['N_r']+0.0) # the support assumed by deconv
    x_ml = ml_info['x']*ml_info['gamma']/(data_info['N_r']+0.0)
    
    ## ml estimation
    err_dd = dist_W1(p,p_hat,x,x_dd)
    err_ml = dist_W1(p,p_hat_ml,x,x_ml)
    
    if len(dd_info['Y_supp'].shape)==1 and verbose:
        # first figure: the confidence interval
        P_model = Pmodel_cal(x,dd_info['Y_supp'],data_info['N_r'],noise=data_info['noise'])
        x_s = np.linspace(0,1,101)
        plt.figure(figsize=[18,5])
        plt.subplot(121)
        plt.plot(x_s,supp_trans(p_hat_ml,x_ml,x_s),linewidth=4,label='ml: %s'%str(err_ml)[0:6],alpha=0.6,color='royalblue')
        plt.plot(x_s,supp_trans(p_hat,x_dd,x_s),linewidth=4,label='dd: %s'%str(err_dd)[0:6],alpha=0.6,color='darkorange')
        plt.plot(x_s,supp_trans(p,x,x_s),linewidth=4,label='true distribution',alpha=0.6,color='seagreen')
        plt.ylim([0,1.2*supp_trans(p,x,x_s).max()])
        plt.legend()
        plt.title('pdf')
        plt.subplot(122)
        plt.plot(x_ml,np.cumsum(p_hat_ml),marker='o',label='ml: %s'%str(err_ml)[0:6],alpha=0.6,color='royalblue')
        plt.plot(x_dd,np.cumsum(p_hat),marker='o',label='dd: %s'%str(err_dd)[0:6],alpha=0.6,color='darkorange')
        plt.plot(x,np.cumsum(p),marker='o',label='true distribution',alpha=0.6,color='seagreen')
        plt.title('cdf')
        plt.legend()
        plt.show()
    return err_dd,err_ml

def supp_trans(p,x,x_new):
    cdf        = np.cumsum(p)
    p_new      = np.interp(x_new,x,cdf)
    p_new[1:] -= p_new[0:-1]
    return p_new

"""
    denoising 
"""
def denoise_1d(Y,data_info=None,noise='poi',opt='dd',lamb=0.1):
    ## if p is not given, using density deconvolution to estimate p
    if data_info is None: _,data_info=dd_1d(Y)

    p     = data_info['p']
    x     = data_info['x']
    gamma=data_info['gamma']  if 'gamma' in data_info.keys() else data_info['N_r']
        
    ## calculate the Bayes estimate of x for each value of y: E[X|Y=y]
    x_hat_dic = {}
    for y in np.unique(Y):
        temp = 0 
        py   = 0
        for i in range(p.shape[0]):
            temp += x[i]*p[i]*sp.stats.poisson.pmf(y,x[i]*gamma)
            py   += p[i]*sp.stats.poisson.pmf(y,x[i]*gamma)
        x_hat_dic[y] = temp/py
        
    ## generates the final results
    Y_hat = np.zeros(Y.shape[0],dtype=float)
    for i in range(Y_hat.shape[0]):
        Y_hat[i] = x_hat_dic[Y[i]]*gamma
    return Y_hat

def denoise_1d_mp(GC,n_job=1,verbose=False,GC_true=None):
    if verbose: 
        print('n_gene=%s, n_cell=%s, n_job=%s'%(str(GC.shape[0]),str(GC.shape[1]),str(n_job)))
        start_time=time.time()
        print('#time start: 0.0s')
    
    Y_input = []
    for i in range(GC.shape[0]):
        Y_input.append(GC[i,:])
        
    if verbose: print('#time input: %ss'%str(time.time()-start_time)[0:5])
    
    ## multi threading
    pool = Pool(n_job)
    res  = pool.map(denoise_1d, Y_input)
    
    if verbose: print('#time mp: %ss'%str(time.time()-start_time)[0:5])
    
    GC_hat = np.zeros(GC.shape)
    for i in range(GC.shape[0]):
        GC_hat[i,:] = res[i]
        
    if verbose: 
        print('#time total: %ss'%str(time.time()-start_time)[0:5])        
        print('MSE: %s \n'%str(np.sum((GC_hat-GC_true)**2)/GC.shape[1]/GC.shape[0]))
                
    return GC_hat


def moment_1d(Y_input):
    data,k,n_sub = Y_input
    M = np.zeros([data.shape[1],k],dtype=float)
    M_ml = np.zeros([data.shape[1],k],dtype=float)
    gene_list = []
    for i_gene,gene in enumerate(data.var_names):
        if data.shape[1]==1:
            Y = np.array(data.X,dtype=int)
        else:
            Y = np.array(data[:,gene].X,dtype=int)
        if n_sub is not None: Y=sub_sample(Y,int(n_sub*Y.sum()))
        gene_list.append(gene)
        _,_,M[i_gene,:],_ = dd_moments_1d(Y,gamma=1)    
        for i in range(k):
            M_ml[i_gene,i] = np.mean(Y**(i+1))
        
    return M,M_ml,gene_list
    
def moment_1d_mp(data,n_job=1,k=2,n_sub=None,verbose=False,GC_true=None):
    

    n_gene,n_cell = data.shape[1],data.shape[0]
    if verbose: 
        print('n_gene=%s, n_cell=%s, n_job=%s'%(str(n_gene),str(n_cell),str(n_job)))
        start_time=time.time()
        print('#time start: 0.0s')
    
    subdata_size = int(n_gene/n_job/3)
    gene_list = list(data.var_names)
    subgene_list = []
    i=0
    while 1:
        if i+subdata_size<len(gene_list):
            subgene_list.append(gene_list[i:i+subdata_size])
            i = i+subdata_size
        else:
            subgene_list.append(gene_list[i:])
            break
    
    Y_input = []
    for subgene in subgene_list:
        Y_input.append([data[:,subgene],k,n_sub])    
    if verbose: print('#time input: %0.4fs'%(time.time()-start_time))
        
    ## multi threading
    pool = Pool(n_job)
    res  = pool.map(moment_1d, Y_input)
    
    if verbose: print('#time mp: %0.4fs'%(time.time()-start_time))
        
    M = np.zeros([n_gene,k],dtype=float)
    M_ml = np.zeros([n_gene,k],dtype=float)
    gene_list = []
    iloc = 0
    for i in range(len(res)):
        subM,subM_ml,subgene = res[i]
        gene_list = gene_list + subgene
        if iloc+subM.shape[0]<n_gene:
            M[iloc:iloc+subM.shape[0],:] = subM
            M_ml[iloc:iloc+subM.shape[0],:] = subM_ml
            iloc = iloc+subM.shape[0]
        else:
            M[iloc:,:] = subM
            M_ml[iloc:,:] = subM_ml
    if verbose: 
        print('#time total: %0.4fs'%(time.time()-start_time))        
                
    return M,M_ml,gene_list

def dd_moment_anndata(data,k=2,verbose=False,size_norm=True):
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    n_cell,n_gene = data.shape
    X = data.X    
    gene_name = list(data.var_names)
    M,M_ml = np.zeros([n_gene,k],dtype=float),np.zeros([n_gene,k],dtype=float)
    
    ## size factor
    if size_norm is True:
        Nrc = X.sum(axis=1)
        Nr = Nrc.mean()
        rsf = Nrc/Nr
    else:
        rsf = np.ones([n_cell],dtype=float)
    rrsf = 1/rsf.clip(min=0.1)
    rrsf = np.array(rrsf,dtype=float)
    
        
    ## moment calculation
    #for i in range(k):
    #    M_ml[:,i] = (X.power(i+1)).mean(axis=0)
       
    ## some hacky implementation
    for i in range(k):
        temp_X = X.power(i+1)
        for i_row in range(n_cell):
            temp = np.squeeze(np.array(temp_X[i_row,:].todense()))
            temp = temp*rrsf[i_row]**(i+1)             
            M_ml[:,i] = M_ml[:,i]+temp
        M_ml[:,i] = M_ml[:,i]/np.sum(rrsf**(i+1))
    
    for i in range(k):
        coef_ = np.poly1d(np.arange(i+1),True).coef
        for j in range(coef_.shape[0]):
            M[:,i] = M[:,i] + coef_[j]*M_ml[:,i-j]  
    
    #if verbose: print('Nr=%d'%Nr)
    #for i in range(k):
    #    M[:,i] = M[:,i]/Nr**(i+1)
    #    M_ml[:,i] = M_ml[:,i]/Nr**(i+1)
    
    if verbose: 
        print('#time total: %0.4fs'%(time.time()-start_time)) 
    return M,M_ml,gene_name