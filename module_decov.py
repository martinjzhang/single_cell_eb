import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvx
import time
from util import *

### testing bench 
#def simu_dd(p,x,param_list,rep_time,output_folder):
#    res_dd=np.zeros([param_list.shape[0],rep_time],dtype=float)
#    res_ml=np.zeros([param_list.shape[0],rep_time],dtype=float)
#    B_list=param_list[:,2]
#    
#    for i in range(param_list.shape[0]):
#        N_c=param_list[i,0]
#        N_r=param_list[i,1]
#        for j in range(rep_time):
#            Y_pdf=data_gen_poi(p,x,N_c,N_r)
#            p_hat_dd,dd_info=dd_1d(Y_pdf,x,N_c,N_r)
#            p_hat_ml=esti_ml(Y_pdf,x,N_c,N_r)
#            if p_hat_dd is not None:
#                res_dd[i,j]=dd_evaluation(p,p_hat_dd)
#            else:
#                print 'error'
#            res_ml[i,j]=dd_evaluation(p,p_hat_ml)
#                
#                
#    ## store the results
#    plt.figure()
#    plt.plot(x,p,marker='o')
#    plt.savefig(output_folder+'/true_dist.png')
#    plt.close()
#    
#    np.save(output_folder+'/res.npy',(res_dd,res_ml,p,x,param_list))
#    
#    err_dd=np.mean(res_dd,axis=1)
#    err_ml=np.mean(res_ml,axis=1)
#    vmin=np.min([np.min(err_dd),np.min(err_ml)])
#    vmax=np.max([np.max(err_dd),np.max(err_ml)])
#    
#    best_param=[]
#    for B in np.unique(B_list):
#        temp=np.argmin(err_dd[B_list==B])
#        best_param.append(list(param_list[B_list==B,:][temp,0:2]))
#    best_param=np.array(best_param)
#    
#    
#    plt.figure(figsize=[20,8])    
#    plt.subplot(1,2,1)
#    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=1000, c=err_dd,cmap='viridis',vmin=0, vmax=0.4)
#    plt.plot(np.log(best_param[:,0]),best_param[:,1],marker='o',color='r')
#    plt.xlabel('log Nc')
#    plt.ylabel('Nr')
#    plt.title('density deconvolution')
#    plt.colorbar()
#    
#    plt.subplot(1,2,2)
#    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=1000, c=err_ml,cmap='viridis',vmin=0, vmax=1)
#    plt.xlabel('log Nc')
#    plt.ylabel('Nr')
#    plt.title('maximum likelihood')
#    plt.colorbar()
#    plt.savefig(output_folder+'/error_plot.png')
#    plt.close()   
#    return res_dd,res_ml
#
### testing bench 
#def simu_dd_2d(p,x,param_list,rep_time,output_folder):
#    res_dd=np.zeros([param_list.shape[0],rep_time],dtype=float)
#    res_ml=np.zeros([param_list.shape[0],rep_time],dtype=float)
#    B_list=param_list[:,2]
#    
#    for i in range(param_list.shape[0]):
#        for j in range(rep_time):
#            N_c=param_list[i,0]
#            N_r=param_list[i,1]
#            while True:                
#                try: 
#                    res = exp_2d(p,x,(N_c,N_r,'mml',0.1,0))
#                    res_dd[i,j]=res['err_tv_dd']
#                    res_ml[i,j]=res['err_tv_ml']
#                    break
#                except:              
#                    print 'exception, do it again!'            
#                
#    ## store the results
#    plt.figure()
#    plot_density_2d(p,x)
#    plt.savefig(output_folder+'/true_dist.png')
#    plt.close()
#    
#    np.save(output_folder+'/res.npy',(res_dd,res_ml,p,x,param_list))
#    
#    err_dd=np.mean(res_dd,axis=1)
#    err_ml=np.mean(res_ml,axis=1)
#    vmin=np.min([np.min(err_dd),np.min(err_ml)])
#    vmax=np.max([np.max(err_dd),np.max(err_ml)])
#    
#    best_param=[]
#    for B in np.unique(B_list):
#        temp=np.argmin(err_dd[B_list==B])
#        best_param.append(list(param_list[B_list==B,:][temp,0:2]))
#    best_param=np.array(best_param)
#    
#    
#    plt.figure(figsize=[20,8])    
#    plt.subplot(1,2,1)
#    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=20000*err_dd**2, c=err_dd,cmap='viridis',vmin=0, vmax=0.4,alpha=0.8)
#    plt.plot(np.log(best_param[:,0]),best_param[:,1],marker='o',color='r')
#    plt.xlabel('log Nc')
#    plt.ylabel('Nr')
#    plt.title('density deconvolution')
#    plt.colorbar()
#    
#    plt.subplot(1,2,2)
#    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=20000*err_ml**2, c=err_ml,cmap='viridis',vmin=0, vmax=1,alpha=0.8)
#    plt.xlabel('log Nc')
#    plt.ylabel('Nr')
#    plt.title('maximum likelihood')
#    plt.colorbar()
#    plt.savefig(output_folder+'/error_plot.png')
#    plt.close()   
#    return res_dd,res_ml
#
#def exp_2d(p,x,param): 
#    # param: the parameter tuple with (N_c,N_r,option,lamb,vis)
#    N_c    = param[0]
#    N_r    = param[1]
#    option = param[2]
#    lamb   = param[3]
#    vis    = param[4]
#    print 'N_c:%s, N_r:%s, option:%s, lamb:%s'%(str(N_c),str(N_r),option,str(lamb))
#    
#    #generating the data 
#    Y_pdf,Y_supp = data_gen_poi_2d(p,x,N_c,N_r)
#    if vis == 1:
#        plt.figure(figsize=[12,5])
#        plt.subplot(1,2,1)
#        plot_density_2d(p,x)
#        plt.subplot(1,2,2)
#        plot_density_2d(Y_pdf,Y_supp)
#        plt.show()
#                              
#    #recover the probability
#    p_hat,dd_info = dd_2d(Y_pdf,Y_supp,x,N_c,N_r,lamb=lamb,option=option)
#    p_hat_ml      = esti_ml(Y_pdf,Y_supp,x,N_c,N_r) 
#    
#    if vis == 1:
#        _=plot_dd_result(p,p_hat,dd_info)
#    
#    res              = {}    
#    res['p']         = p
#    res['x']         = x
#    res['p_hat']     = p_hat
#    res['p_hat_ml']  = p_hat_ml
#    res['true_pc']   = pearson_corr(p,x)
#    res['true_mi']   = mutual_info(p,x)
#    res['dd_pc']     = pearson_corr(p_hat,x)
#    res['dd_mi']     = mutual_info(p_hat,x)
#    res['ml_pc']     = pearson_corr(p_hat_ml,x)
#    res['ml_mi']     = mutual_info(p_hat_ml,x)    
#    res['err_tv_dd'] = dist_tv(p,p_hat)
#    res['err_tv_ml'] = dist_tv(p,p_hat_ml)
#    res['err_pc_dd'] = np.absolute((res['dd_pc']-res['true_pc'])/res['true_pc']*100)
#    res['err_pc_ml'] = np.absolute((res['ml_pc']-res['true_pc'])/res['true_pc']*100)
#    res['err_mi_dd'] = np.absolute((res['dd_mi']-res['true_mi'])/res['true_mi']*100)
#    res['err_mi_ml'] = np.absolute((res['ml_mi']-res['true_mi'])/res['true_mi']*100)
#    
#    return res
    
## toy distribution
def toy_dist(opt='1d',vis=0):
    if opt == '1d':
        p = np.array([0.2,0.3,0.4,0.1,0,0,0,0,0,0])
        x = np.linspace(0,0.9,10)
        if vis == 1:
            plt.figure()
            plot_density_1d(p,x)
            plt.xlabel('support')
            plt.ylabel('probablity')
            plt.title('toy distribution')
            plt.legend()
            plt.show()
    elif opt == '2d':
        pass
    else: 
        print 'Err: option not recognized!'
    return p,x

## data generation
def data_gen_1d(p,x,N_c,N_r,return_data=0,noise='poi'):
    n_supp = p.shape[0]
    x_samp = np.random.choice(x,N_c, p=p,replace=True)
    if noise == 'poi':
        Y=np.random.poisson(x_samp*N_r)
    elif noise == 'bin':
        Y=np.random.binomial(N_r,x_samp)
        
    Y_pdf=np.bincount(Y)
    Y_pdf=Y_pdf/(Y.shape[0]+0.0)
    Y_supp=np.arange(Y_pdf.shape[0])
    
    if return_data == 0:
        return Y_pdf,Y_supp
    else:
        return Y_pdf,Y_supp,x_samp,Y

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
def deconv_1d(Y_pdf,Y_supp,x,N_c,N_r,noise='poi',opt='dd',lamb=0.1):
    ## parameter setting    
    n_supp=x.shape[0]
    n_obs=Y_pdf.shape[0]
    delta=1e-2
    #min_clamp=0.01
    
    ## calculate the noise channel matrix, P_model: n_obs * n_supp
    if noise == 'poi':
        P_model=sp.stats.poisson.pmf(np.repeat(np.reshape(np.arange(n_obs),[n_obs,1]),n_supp,axis=1),\
                                    np.repeat(np.reshape(x*N_r,[1,n_supp]),n_obs,axis=0))
    elif noise == 'bin':
        P_model=sp.stats.binom.pmf(np.repeat(np.reshape(np.arange(n_obs),[n_obs,1]),n_supp,axis=1),N_r,\
                                    np.repeat(np.reshape(x,[1,n_supp]),n_obs,axis=0))
    else: 
        print 'Err: noise type not recognized!'
        return
    
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
    p_hat=np.array(p_hat_cvx.value).flatten() 
        
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
    return p_hat,dd_info

def denoise_1d(Y,N_r,x,p=None,noise='poi',opt='dd',lamb=0.1):
    ## if p is not given, using density deconvolution to estimate p
    if p is None: 
        N_c    = Y.shape[0]
        Y_pdf  = np.bincount(Y)/(N_c+0.0)
        Y_supp = np.arange(Y_pdf.shape[0])
        p,_    = deconv_1d(Y_pdf,Y_supp,x,N_c,N_r,noise=noise,opt=opt,lamb=lamb)
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
    return x_hat

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


def esti_ml(Y_pdf,Y_supp,x,N_c,N_r):
    n_obs=Y_pdf.shape[0]
    Y_supp=Y_supp/(N_r+0.0)
    if len(Y_supp.shape)==1:
        Y_supp=np.reshape(Y_supp,[n_obs,1])
        x=np.reshape(x,[x.shape[0],1])
    p_hat=np.zeros(x.shape[0],dtype=float)
    for i in range(Y_supp.shape[0]):
        p_hat[np.argmin(np.linalg.norm(x-Y_supp[i,:],axis=1))]+=Y_pdf[i]
    return p_hat

def dd_evaluation(p,p_hat):
    return np.linalg.norm(p_hat-p,ord=1)
   

## visualization of the distribution estimation result
def plot_dd_result(p,p_hat,dd_info):
    p_hat_ml=esti_ml(dd_info['Y_pdf'],dd_info['Y_supp'],dd_info['x'],dd_info['N_c'],dd_info['N_r'])        
    err_dd=np.linalg.norm(p_hat-p,ord=1)
    err_ml=np.linalg.norm(p_hat_ml-p,ord=1)
    
    if len(dd_info['Y_supp'].shape)==1:
        plt.figure(figsize=[12,5])
        plt.subplot(1,2,1)
        plt.plot(np.log(dd_info['P_model'].dot(p)+0.01),label='true value')
        plt.plot(np.log(dd_info['P_model'].dot(p_hat)+0.01),label='recovered value')
        plt.plot(np.log(dd_info['Y_pdf_ub']+0.01),label='upper bound')
        plt.plot(np.log(dd_info['Y_pdf_lb']+0.01),label='lower bound')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(dd_info['x'],p_hat_ml,marker='o',label='ML: %s'%str(err_ml)[0:6])
        plt.plot(dd_info['x'],p_hat,marker='o',label='density decov: %s'%str(err_dd)[0:6])
        plt.plot(dd_info['x'],p,marker='o',label='true distribution')
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
