import numpy as np
import scipy as sp
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvx
import time
from util import *
    
## toy distribution
def toy_dist(opt='1d',vis=0):
    if opt == '1d':
        p = np.array([0,0.2,0.3,0.4,0.1,0,0,0,0,0])
        x = np.array([0,0.05,0.15,0.25,0.35,0.4,0.5,0.6,0.7,1.0])
        if vis == 1:
            plt.figure()
            plot_density_1d(p,x)
            plt.xlabel('support')
            plt.ylabel('probablity')
            plt.title('toy distribution')
            plt.legend()
            plt.show()
    elif opt == 'test_case1':
        p = np.array([0,1,0,0.1,0.02,0,0.01,0,0.02,0.05])
        p /= np.sum(p)
        x = np.array([0,0.05,0.15,0.25,0.35,0.4,0.5,0.6,0.7,1.0])
    elif opt == '2d':
        pass
    else: 
        print 'Err: option not recognized!'
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
        plt.hist(x_samp)
        plt.subplot(122)
        plt.hist(Y)
        plt.show()
    return x_samp,Y,data_info
    #Y_pdf=np.bincount(Y)
    #Y_pdf=Y_pdf/(Y.shape[0]+0.0)
    #Y_supp=np.arange(Y_pdf.shape[0])
    #
    #if return_data == 0:
    #    return Y_pdf,Y_supp
    #else:
    #    return Y_pdf,Y_supp,x_samp,Y

def data_gen_poi_2d(p,x,N_c=1000,N_r=10):
    #start=time.time()
    #print time.time()-start
    n_supp=p.shape[0]
    x=x*N_r
    idx_select=np.random.choice(np.arange(n_supp),N_c, p=p,replace=True)
    Y=np.zeros([N_c,2],dtype=int)
    Y[:,0]=np.random.poisson(x[idx_select,0])
    Y[:,1]=np.random.poisson(x[idx_select,1])
    Y_supp=[]
    #print time.time()-start
    for i in range(Y[:,0].min(),Y[:,0].max()+1):
        for j in range(Y[:,1].min(),Y[:,1].max()+1):
            Y_supp.append([i,j])
    #print len(Y_supp), Y_supp[0], Y_supp[-1]
    Y_supp=np.array(Y_supp)
    Y_pdf=np.zeros(Y_supp.shape[0],dtype=float)
    #print time.time()-start
    Y_unique,idx_reverse=np.unique(Y,return_inverse=True,axis=0)
    Y_freq=np.bincount(idx_reverse)
    
    Y_supp_tuple=[tuple(ele) for ele in Y_supp]
    Y_unique_tuple=[tuple(ele) for ele in Y_unique]
    temp_idx=[Y_supp_tuple.index(i) for i in Y_unique_tuple]
    Y_pdf[temp_idx]=Y_freq
    #print time.time()-start
    Y_pdf=Y_pdf/(N_c+0.0)
    #print time.time()-start
    return Y_pdf,Y_supp

def hist_2d(Y):
    print '------ hist_2d start'
    start=time.time()
    print time.time()-start
    N_c=Y.shape[0]
    Y_supp=[]
    print Y[:,0].min(),Y[:,0].max()+1
    for i in range(Y[:,0].min(),Y[:,0].max()+1):
        for j in range(Y[:,1].min(),Y[:,1].max()+1):
            Y_supp.append([i,j])
    print time.time()-start
    Y_supp=np.array(Y_supp)
    Y_pdf=np.zeros(Y_supp.shape[0],dtype=float)
    Y_unique,idx_reverse=np.unique(Y,return_inverse=True,axis=0)
    Y_freq=np.bincount(idx_reverse)  
    print time.time()-start
    Y_supp_tuple=[tuple(ele) for ele in Y_supp]
    Y_unique_tuple=[tuple(ele) for ele in Y_unique]
    temp_idx=[Y_supp_tuple.index(i) for i in Y_unique_tuple]
    print time.time()-start
    Y_pdf[temp_idx]=Y_freq
    Y_pdf=Y_pdf/(N_c+0.0)
    print time.time()-start
    print '------ hist_2d end'
    return Y_pdf,Y_supp

## estimation methods 
def counts2pdf_1d(Y):
    Y_pdf=np.bincount(Y)
    Y_pdf=Y_pdf/(Y.shape[0]+0.0)
    Y_supp=np.arange(Y_pdf.shape[0])
    return Y_pdf,Y_supp

def cal_Nr(Y): # we use this function to define the data driven N_r for a
    N_r = np.percentile(Y,99) # N_r should be roughly Y_max. The 95% percentile is used for robustness consideration
    return N_r

#def esti_ml(Y_pdf,Y_supp,x,N_c,N_r): # with no rounding to the nearest neighbour
def ml_1d(Y): # with no rounding to the nearest neighbour
    Y_pdf,Y_supp = counts2pdf_1d(Y)
    N_c          = Y.shape[0]
    N_r          = cal_Nr(Y)
    p_hat        = Y_pdf
    x            = Y_supp/(N_r+0.0)
    
    # recording the information
    ml_info={}
    ml_info['Y_pdf']  = Y_pdf
    ml_info['Y_supp'] = Y_supp
    ml_info['x']      = x
    ml_info['N_c']    = N_c
    ml_info['N_r']    = N_r
    return p_hat,x,ml_info

#def deconv_1d(Y,x,N_c,N_r,noise='poi',opt='dd',lamb=0.1):
def deconv_1d(Y,noise='poi',opt='dd',lamb=0.1,n_xsupp=11):
    ## converting the read counts to some sufficient statistics
    Y_pdf,Y_supp = counts2pdf_1d(Y)
    
    ## parameter setting   
    N_c    = Y.shape[0]
    N_r    = cal_Nr(Y)
    x      = np.linspace(0,1,n_xsupp)
    n_supp = x.shape[0]
    n_obs  = Y_supp.shape[0]
    delta  = 1e-2
    
    ## calculate the noise channel matrix, P_model: n_obs * n_supp
    P_model = Pmodel_cal(x,Y_supp,N_r,noise=noise)
    
    ## calculating the confidence interval
    Y_pdf_ub = np.zeros([n_obs],dtype=float)
    Y_pdf_lb = np.zeros([n_obs],dtype=float)
    for i in range(n_obs):
        Y_pdf_lb[i],Y_pdf_ub[i] = binomial_ci_cp(int(Y_pdf[i]*N_c),N_c,delta/(n_obs+0.0))
        
    #for i in range(n_obs):
    #    print Y_pdf[i],Y_pdf_lb[i],Y_pdf_ub[i]    
    #return
    
    ## linear programming
    p_hat_cvx = cvx.Variable(n_supp)
    if opt == 'dd': # vanilla density deconvolution        
        objective=cvx.Minimize(1)
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]  
    elif opt == 'dd_lip': # density deconvolution with Lipschitz regularization
        A=np.zeros([n_supp,n_supp],dtype=float)
        for i in range(n_supp):
            x_min=np.partition(np.absolute(x-x[i]),1)[1]
            for j in range(i+1,n_supp):
                if np.absolute(x[j]-x[i])==x_min:
                    A[i,i]=A[i,i]+1
                    A[j,j]+=1
                    A[i,j]-=1
                    A[j,i]-=1
        objective=cvx.Minimize(cvx.norm(Y_pdf-P_model*p_hat_cvx)+lamb*cvx.quad_form(p_hat_cvx,A))
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]
    else: 
        print 'Err: deconv method not recognized'    
    prob = cvx.Problem(objective, constraints)
    prob.solve()

    ## if the constraints are not satisfied, using mml:
    if prob.status != 'optimal':
        print 'Constraints are not satisfied; solving instead using '+opt+'_mml'
        if opt == 'dd':
            objective=cvx.Minimize(-Y_pdf.T*cvx.log(P_model*p_hat_cvx))
        elif opt == 'dd_lip':
            objective=cvx.Minimize(-Y_pdf.T*cvx.log(P_model*p_hat_cvx)+lamb*cvx.quad_form(p_hat_cvx,A))
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1] 
        prob = cvx.Problem(objective, constraints)
        prob.solve()
    
    ## output the result
    p_hat = np.array(p_hat_cvx.value).flatten().clip(min=0)
    p_hat/= np.sum(p_hat)
        
    ## other informations 
    dd_info = {}
    dd_info['Y_pdf']    = Y_pdf
    dd_info['Y_supp']   = Y_supp
    dd_info['Y_pdf_ub'] = Y_pdf_ub
    dd_info['Y_pdf_lb'] = Y_pdf_lb   
    dd_info['P_model']  = P_model
    dd_info['x']        = x
    dd_info['N_c']      = N_c
    dd_info['N_r']      = N_r    
    return p_hat,x,dd_info

# functions called by deconv_1d
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
        print 'Err: noise type not recognized!'
        return
    return P_model
    
def binomial_ci_cp(k,n,delta):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    Modified from https://gist.github.com/DavidWalz/8538435
    """
    if k==0:
        lb=0
        ub=(1-(delta/2.0)**(1/(n+0.0)))
    elif k==n:
        lb=(delta/2.0)**(1/(n+0.0))
        ub=1
    else:
        lb = sp.stats.beta.ppf(delta/2, k, n-k+1)
        ub = sp.stats.beta.ppf(1-delta/2, k+1, n-k)
    return max(lb,0), min(ub,1)

#def dd_moments_1d(Y_pdf,Y_supp,x,N_c,N_r,k=2,noise='poi'):
def dd_moments_1d(Y,k=2,noise='poi'):
    ## converting the read counts to some sufficient statistics
    Y_pdf,Y_supp = counts2pdf_1d(Y)
    ## parameter setting   
    N_c    = Y.shape[0]
    N_r    = cal_Nr(Y)
    
    M_hat = np.zeros(k)   
    if noise == 'poi':
        for i in range(k):
            for j in range(Y_supp.shape[0]):
                temp = 1
                for l in range(i+1):
                    temp *= Y_supp[j]-l
                M_hat[i] += Y_pdf[j] * temp
            M_hat[i] /= (N_r**(i+1)+0.0)
    if noise == 'bin':
        for i in range(k):
            for j in range(Y_supp.shape[0]):
                if Y_supp[j] >= i+1:
                    M_hat[i] += Y_pdf[j]*sp.special.comb(Y_supp[j], i+1)/(sp.special.comb(N_r, i+1)+0.0)
    mean_hat = M_hat[0]
    var_hat  = M_hat[1]-M_hat[0]**2
    return mean_hat,var_hat,M_hat,N_r
def M_convert(M,N_r1,N_r2):
    M2 = np.zeros(M.shape,dtype=float)
    r  = N_r1/(N_r2+0.0)
    for i in range(M2.shape[0]):
        M2[i] = M[i]*r**(i+1)
    mean2 = M2[0]
    var2  = M2[1]-M2[0]**2
    return mean2,var2,M2

#def denoise_1d(Y,N_r,x,p=None,noise='poi',opt='dd',lamb=0.1):
def denoise_1d(Y,data_info=None,noise='poi',opt='dd',lamb=0.1):
    ## if p is not given, using density deconvolution to estimate p
    if data_info is None: 
        N_c    = Y.shape[0]
        Y_pdf  = np.bincount(Y)/(N_c+0.0)
        Y_supp = np.arange(Y_pdf.shape[0])
        #p,_    = deconv_1d(Y_pdf,Y_supp,x,N_c,N_r,noise=noise,opt=opt,lamb=lamb)
        p,x,dd_info = deconv_1d(Y,noise=noise,opt=opt,lamb=lamb)
        N_r    = dd_info['N_r']
    else: 
        p   = data_info['p']
        x   = data_info['x']
        N_r = data_info['N_r'] 
        
    ## calculate the Bayes estimate of x for each value of y: E[X|Y=y]
    x_hat_dic = {}
    for y in np.unique(Y):
        temp = 0 
        py   = 0
        for i in range(p.shape[0]):
            temp += x[i]*p[i]*sp.stats.poisson.pmf(y,x[i]*N_r)
            py   += p[i]*sp.stats.poisson.pmf(y,x[i]*N_r)
        x_hat_dic[y] = temp/py
        
    ## generates the final results
    x_hat = np.zeros(Y.shape[0],dtype=float)
    for i in range(x_hat.shape[0]):
        x_hat[i] = x_hat_dic[Y[i]]
    return x_hat,N_r

def GC_convert(x1,N_r1,N_r2):
    return x1*N_r1/(N_r2+0.0)
def dd_2d(Y_pdf,Y_supp,x,N_c,N_r,option='mml',lamb=0.1):
    print '------ dd_2d start, option:%s, lambda:%s'%(option,str(lamb))
    start  = time.time()
    n_supp = x.shape[0]
    n_obs  = Y_pdf.shape[0]
    
    delta=1e-3
    P_model_1=sp.stats.poisson.pmf(np.repeat(np.reshape(Y_supp[:,0],[n_obs,1]),n_supp,axis=1),\
                                np.repeat(np.reshape(x[:,0]*N_r,[1,n_supp]),n_obs,axis=0))
    P_model_2=sp.stats.poisson.pmf(np.repeat(np.reshape(Y_supp[:,1],[n_obs,1]),n_supp,axis=1),\
                                np.repeat(np.reshape(x[:,1]*N_r,[1,n_supp]),n_obs,axis=0))
    P_model=P_model_1*P_model_2
    print '------ model cal completed: %s' %str(time.time()-start)

    Y_pdf_lb=np.zeros([n_obs],dtype=float)
    Y_pdf_ub=np.zeros([n_obs],dtype=float)
    for i in range(n_obs):
        Y_pdf_lb[i],Y_pdf_ub[i]=binomial_ci_cp(Y_pdf[i]*N_c,N_c,delta)
  
    print '------ optimization start: %s' %str(time.time()-start)
    ## linear programming
    p_hat_cvx=cvx.Variable(n_supp)
    ## minimize the continuity
    A=np.zeros([n_supp,n_supp],dtype=float)
    for i in range(n_supp):
        x_min=np.partition(np.linalg.norm(x-x[i,:],axis=1),1)[1]
        for j in range(i+1,n_supp):
            if np.linalg.norm(x[j,:]-x[i,:])==x_min:
                A[i,i]=A[i,i]+1
                A[j,j]+=1
                A[i,j]-=1
                A[j,i]-=1    
    
    ## optimization
    ## 1: simple density deconvolution 
    #objective=cvx.Minimize(1)
    #objective=cvx.Minimize(cvx.norm(Y_pdf-P_model*p_hat_cvx))
    #constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]
    
    ## 2: density deconvolution with continuity regularization 
    #objective=cvx.Minimize(1)
    #constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]
    
    ## 3: maximum likelihood 
    #objective=cvx.Minimize(-Y_pdf.T*cvx.log(P_model*p_hat_cvx)+0.1*cvx.quad_form(p_hat_cvx,A))
    #constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1]       
    
    if option   == 'dd':
        ## density deconvolution with continuity regularization 
        objective=cvx.Minimize(cvx.norm(Y_pdf-P_model*p_hat_cvx)+lamb*cvx.quad_form(p_hat_cvx,A))
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]
    elif option == 'mml':
        ## multiset maximum likelihood 
        objective=cvx.Minimize(-Y_pdf.T*cvx.log(P_model*p_hat_cvx)+lamb*cvx.quad_form(p_hat_cvx,A))
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1]  
    elif option == 'debug':
        objective=cvx.Minimize(-Y_pdf.T*cvx.log(P_model*p_hat_cvx)+lamb*cvx.quad_form(p_hat_cvx,A))
        constraints=[p_hat_cvx>=0,cvx.sum_entries(p_hat_cvx)==1,P_model*p_hat_cvx<=Y_pdf_ub,P_model*p_hat_cvx>=Y_pdf_lb]
    else: 
        print 'option is not valid'
        return  
    
    ## solver
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    p_hat=np.array(p_hat_cvx.value).flatten()
    
    print '------ optimization end: %s' %str(time.time()-start)
    print 'KL:%s, TV:%s, continuity: %s'%(str(dist_kl(Y_pdf,P_model.dot(p_hat)))[0:6],\
                                          str(dist_tv(Y_pdf,P_model.dot(p_hat)))[0:6],
                                          str(p_hat.dot(A).dot(p_hat))[0:6])
    
    if p_hat[0] is None:
        print '------ optimization failed!'
        return 
    
    Y_pdf_hat=P_model.dot(p_hat)
    #for i in range(Y_supp.shape[0]):
    #    print Y_supp[i,:], Y_pdf_hat[i], Y_pdf[i], Y_pdf_lb[i],Y_pdf_ub[i]
    
    ## other informations 
    dd_info={}
    dd_info['Y_pdf']=Y_pdf
    dd_info['Y_supp']=Y_supp
    dd_info['Y_pdf_ub']=Y_pdf_ub
    dd_info['Y_pdf_lb']=Y_pdf_lb   
    dd_info['P_model']=P_model
    dd_info['x']=x
    dd_info['N_c']=N_c
    dd_info['N_r']=N_r
    
    print '------ dd_2d end: %s'%str(time.time()-start)
    return p_hat, dd_info

#def binomial_ci_old(Y,N_c,delta):
#    min_clamp=1/(N_c+0.0)
#    p_hat=Y/(N_c+0.0)
#    p_hat=p_hat.clip(min=min_clamp)  
#    epsilon=np.sqrt(3*np.log(2/delta)/N_c/p_hat.clip(min=min_clamp))
#    ci=epsilon*p_hat.clip(min=min_clamp)
#    ub=(p_hat+ci).clip(max=1)
#    lb=(p_hat-ci).clip(min=0)
#    return lb, ub
#
#def binomial_ci(Y,N_c,delta):
#    p_hat=Y/(N_c+0.0)
#    if Y>0: # large probability regime  
#        epsilon_l=np.sqrt(2*np.log(2/delta)/N_c/p_hat)
#        epsilon_u=np.sqrt(3*np.log(2/delta)/N_c/p_hat)
#        lb=p_hat*(1-epsilon_l)
#        ub=p_hat*(1+epsilon_u)
#    elif Y==0:
#        lb=0
#        ub=(1-(delta/2.0)**(1/(N_c+0.0)))
#    return max(lb,0), min(ub,1)

def dd_evaluation(p,p_hat):
    return np.linalg.norm(p_hat-p,ord=1)
   

## visualization of the distribution estimation result
def plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info):
    ## load some important parameters
    x    = data_info['x'] # the true support 
    x_dd = dd_info['x']*dd_info['N_r']/(data_info['N_r']+0.0) # the support assumed by deconv
    x_ml = ml_info['x']*dd_info['N_r']/(data_info['N_r']+0.0)
    
    ## ml estimation
    #p_hat_ml,x_ml = esti_ml(dd_info['Y_pdf'],dd_info['Y_supp'],dd_info['x'],dd_info['N_c'],dd_info['N_r'])        
    #err_dd=np.linalg.norm(p_hat-p,ord=1)
    #err_ml=np.linalg.norm(p_hat_ml-p,ord=1)
    err_dd = dist_W1(p,p_hat,x,x_dd)
    err_ml = dist_W1(p,p_hat_ml,x,x_ml)
    
    if len(dd_info['Y_supp'].shape)==1:
        # first figure: the confidence interval
        P_model = Pmodel_cal(x,dd_info['Y_supp'],data_info['N_r'],noise=data_info['noise'])
        plt.figure(figsize=[12,5])
        plt.subplot(1,2,1)
        plt.plot(dd_info['Y_supp'],np.log(P_model.dot(p)+0.001),label='true value')
        plt.plot(dd_info['Y_supp'],np.log(dd_info['P_model'].dot(p_hat)+0.001),label='recovered value')
        plt.plot(dd_info['Y_supp'],np.log(dd_info['Y_pdf_ub']+0.001),label='upper bound')
        plt.plot(dd_info['Y_supp'],np.log(dd_info['Y_pdf_lb']+0.001),label='lower bound')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(x_ml,np.cumsum(p_hat_ml),marker='o',label='ml: %s'%str(err_ml)[0:6],alpha=0.6)
        plt.plot(x_dd,np.cumsum(p_hat),marker='o',label='deconv: %s'%str(err_dd)[0:6],alpha=0.6)
        plt.plot(x,np.cumsum(p),marker='o',label='true distribution',alpha=0.6)
        plt.legend()
        plt.show()
    elif len(dd_info['Y_supp'].shape)==2:
        true_pc = pearson_corr(p,dd_info['x'])
        true_mi = mutual_info(p,dd_info['x'])
        dd_pc   = pearson_corr(p_hat,dd_info['x'])
        dd_mi   = mutual_info(p_hat,dd_info['x'])
        ml_pc   = pearson_corr(p_hat_ml,dd_info['x'])
        ml_mi   = mutual_info(p_hat_ml,dd_info['x'])
        
        plt.figure(figsize=[18,5])
        plt.subplot(1,3,1)
        plt.scatter(dd_info['x'][:,0], dd_info['x'][:,1],s=5000*p,alpha=0.8,c=p,cmap='viridis')
        plt.colorbar()
        plt.title('True, PC: %s,  MI: %s'%(str(true_pc)[0:6],str(true_mi)[0:6]))
        plt.subplot(1,3,2)
        plt.scatter(dd_info['x'][:,0], dd_info['x'][:,1],s=5000*p_hat,alpha=0.8,c=p_hat,cmap='viridis')
        plt.colorbar()
        plt.title('Density Deconv TV:%s,\n PC: %s,  error: %s%%,\n MI: %s,  error: %s%%'%(str(err_dd)[0:6],str(dd_pc)[0:6],\
                                                  str(np.absolute((dd_pc-true_pc)/true_pc*100))[0:4],\
                                                str(dd_mi)[0:6],str(np.absolute((dd_mi-true_mi)/true_mi*100))[0:4]))
        plt.subplot(1,3,3)
        plt.scatter(dd_info['x'][:,0], dd_info['x'][:,1],s=5000*p_hat_ml,alpha=0.8,c=p_hat_ml,cmap='viridis')
        plt.colorbar()
        plt.title('ML TV:%s,\n PC: %s,  error: %s%%,\n MI:%s,  error: %s%%,'%(str(err_ml)[0:6],str(ml_pc)[0:6],\
                                                str(np.absolute((ml_pc-true_pc)/true_pc*100))[0:4],
                                                str(ml_mi)[0:6],str(np.absolute((ml_mi-true_mi)/true_mi*100))[0:4]))
        plt.show()
    return err_dd,err_ml
