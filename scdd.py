""" 
    a self-contained module for moment estimation only
    some nice figure generation functions are included
    the data are assumed to be stored with an anndata format
    
    scdd refers to "Single Cell Density Deconvolution"
    
"""

import numpy as np
import scipy as sp
import pandas as pd
import itertools
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import time
from util import *
import seaborn as sns
from BsplineND import *
from multiprocessing import Pool
import time
from sklearn.linear_model import LinearRegression
import scanpy.api as sc
from collections import defaultdict
from collections import Counter

""" 
    calculate the size factor
"""
def sf(data,verbose=True):
    X=data.X
    Nrc = X.sum(axis=1)
    Nr = Nrc.mean()
    
    gamma_c = np.array(Nrc/Nr,dtype=float) 
    
    if verbose: 
        print('Nr=%d'%Nr)
        print('gamma_c',np.percentile(gamma_c,[0,0.01,0.1,99,99.9,100]))
        plt.figure(figsize=[12,5])
        plt.hist(gamma_c,bins=20)
        plt.show()
    return gamma_c

""" 
    calculate the size factor using moment method (maybe not so important since the poisson rate is large anyway)
"""
def sf_m(data,verbose=True):
    pass

"""
    the subsample function for anndata input
    
    ## fixit: the randomness is gone
"""
def subsample_anndata(adata,Nr_new,Nc_new,random_state=0,verbose=True):
    np.random.seed(random_state)
    
    adata = adata.copy()
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    
    ## sub-sample the cells
    Nc,G=adata.shape
    if verbose: print('before cell subsamp',adata.shape)
    sc.pp.subsample(adata,Nc_new/Nc,random_state)
    Nc,G=adata.shape
    if verbose: 
        print('after cell subsamp',adata.shape)
        print('#time sub-sample cells: %0.4fs'%(time.time()-start_time)) 
        
        
    
    ## sub-sample the counts for each cell        
    Nr = adata.X.sum()/Nc
    downsample_rate = Nr_new/Nr  
    Nrc = np.array(adata.X.sum(axis=1)).reshape(-1)
    
    ## change it to np array maybe 
    data_new = []
    indices_new = []
    indptr_new = [0]
    
    for icell, _ in enumerate(adata.obs_names):
        target_Nrc = int(Nrc[icell]*downsample_rate)
        if target_Nrc>0:
            ## subsample 
            idx_vec = []
            idx_gene = []
            temp = adata.X[icell].astype(int)
            
            for val,i_gene in zip(temp.data,temp.indices):
                idx_gene.append(i_gene)
                idx_vec.extend([i_gene]*val)
            
            downsamp = np.random.choice(idx_vec,target_Nrc,replace=False)
            indices,values = np.unique(downsamp,return_counts=True)
                       
            ## set row value
            data_new.extend(list(values))
            indices_new.extend(list(indices))
            indptr_new.append(values.shape[0]+indptr_new[-1])
            
        else:
            indptr_new.append(indptr_new[-1])
        
        if verbose and icell%5000==0 and icell>0:
            print('## %d cells processed'%icell)
    ## construct a new adata file
    X_new = sp.sparse.csr_matrix((data_new, indices_new, indptr_new), shape=adata.shape)
    #print(X_new.todense())
    adata.X = X_new
        
    if verbose: print('#time sub-sample counts: %0.4fs\n'%(time.time()-start_time)) 
    return adata

def set_row_csr(X,i_row,):
    pass
    
    
""" 
    calculate the moments using 1. ml, 2. dd
"""
def dd_moment_anndata(data,k=2,gamma_c=None,verbose=True):
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    X = data.X ## csr file
    gene_name = list(data.var_names)
    M,M_ml = np.zeros([G,k],dtype=float),np.zeros([G,k],dtype=float)
    if gamma_c is None: gamma_c=np.ones([Nc],dtype=float)
        
    ## moment calculation
    for i in range(k):
        M_ml[:,i] = (X.power(i+1)).mean(axis=0)
    
    for i in range(k):
        coef_ = np.poly1d(np.arange(i+1),True).coef
        for j in range(coef_.shape[0]):
            M[:,i] = M[:,i] + coef_[j]*M_ml[:,i-j]  
    
    if verbose: 
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, before size factor')
    
    # adjustion by size factor 
    for i in range(k):   
        gamma_k = (gamma_c**(i+1)).mean()       
        M[:,i] = M[:,i]/gamma_k
        M_ml[:,i] = M_ml[:,i]/gamma_k     
        if verbose: 
            print('M%d, sf=%0.4f'%(i+1,gamma_k))
        
    if verbose: 
        print('#time total: %0.4fs\n'%(time.time()-start_time)) 
        
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, after size factor')
        
        fig_var_mean(M,title='dd log10_var vs log10_mean, after size factor')
        
        fig_M2M1sqr(M,title='dd log10_M2 vs log10_mean^2, after size factor')

    return M,M_ml,gene_name

""" 
    calculate the PC (Pearson correlation) using ml and dd
"""
def dd_PC_anndata(data,gamma_c=None,verbose=True):
    k=2
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    X = data.X ## csr file
    gene_name = list(data.var_names)
    M = np.zeros([G,k],dtype=float)
    M_ml = np.zeros([G,k],dtype=float)
    
    if gamma_c is None: gamma_c=np.ones([Nc],dtype=float)
        
    ## moment calculation
    for i in range(k):
        M_ml[:,i] = (X.power(i+1)).mean(axis=0)
        
    M11_ml = np.array((X.transpose().dot(X)/Nc).todense())
    
    for i in range(k):
        coef_ = np.poly1d(np.arange(i+1),True).coef
        for j in range(coef_.shape[0]):
            M[:,i] = M[:,i] + coef_[j]*M_ml[:,i-j]  
            
    
    if verbose: 
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, before size factor')
    
    # adjustion by size factor 
    for i in range(k):   
        gamma_k = (gamma_c**(i+1)).mean()       
        M[:,i] = M[:,i]/gamma_k
        M_ml[:,i] = M_ml[:,i]/gamma_k    
        
        if k==2:
            M11_ml = M11_ml/gamma_k
        if verbose: 
            print('M%d, sf=%0.4f'%(i+1,gamma_k))
      
    # calculating the PC   
    temp_mu = M_ml[:,0].reshape(G,1)
    temp_sig = np.sqrt((M[:,1]-M[:,0]**2).clip(min=1e-12,max=1e6)).reshape(G,1)
    temp_sig_ml = np.sqrt((M_ml[:,1]-M_ml[:,0]**2).clip(min=1e-12)).reshape(G,1)
    PC = (M11_ml-temp_mu.dot(temp_mu.T))/(temp_sig.dot(temp_sig.T))
    PC_ml = (M11_ml-temp_mu.dot(temp_mu.T))/(temp_sig_ml.dot(temp_sig_ml.T))
    
    PC = PC.clip(max=1,min=-1)
    PC_ml = PC_ml.clip(max=1,min=-1)    
    if verbose: 
        print('#time total: %0.4fs\n'%(time.time()-start_time)) 
        
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, after size factor')
        
        fig_var_mean(M,title='dd log10_var vs log10_mean, after size factor')
        
        fig_M2M1sqr(M,title='dd log10_M2 vs log10_mean^2, after size factor')

    return PC,PC_ml,gene_name

"""

"""
def dd_PC_anndata(data,gamma_c=None,verbose=True):
    k=2
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    X = data.X ## csr file
    gene_name = list(data.var_names)
    M,M_ml = np.zeros([G,k],dtype=float),np.zeros([G,k],dtype=float)
    
    if gamma_c is None: gamma_c=np.ones([Nc],dtype=float)
        
    ## moment calculation
    for i in range(k):
        M_ml[:,i] = (X.power(i+1)).mean(axis=0)
        
    M11_ml = np.array((X.transpose().dot(X)/Nc).todense())
    
    for i in range(k):
        coef_ = np.poly1d(np.arange(i+1),True).coef
        for j in range(coef_.shape[0]):
            M[:,i] = M[:,i] + coef_[j]*M_ml[:,i-j]  
            
    
    if verbose: 
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, before size factor')
    
    # adjustion by size factor 
    for i in range(k):   
        gamma_k = (gamma_c**(i+1)).mean()       
        M[:,i] = M[:,i]/gamma_k
        M_ml[:,i] = M_ml[:,i]/gamma_k    
        
        if k==2:
            M11_ml = M11_ml/gamma_k
        if verbose: 
            print('M%d, sf=%0.4f'%(i+1,gamma_k))
      
    # calculating the PC   
    temp_mu = M_ml[:,0].reshape(G,1)
    temp_sig = np.sqrt((M[:,1]-M[:,0]**2).clip(min=1e-12,max=1e6)).reshape(G,1)
    temp_sig_ml = np.sqrt((M_ml[:,1]-M_ml[:,0]**2).clip(min=1e-12)).reshape(G,1)
    PC = (M11_ml-temp_mu.dot(temp_mu.T))/(temp_sig.dot(temp_sig.T))
    PC_ml = (M11_ml-temp_mu.dot(temp_mu.T))/(temp_sig_ml.dot(temp_sig_ml.T))
    
    PC = PC.clip(max=1,min=-1)
    PC_ml = PC_ml.clip(max=1,min=-1)    
    if verbose: 
        print('#time total: %0.4fs\n'%(time.time()-start_time)) 
        
        fig_var_mean(M_ml,title='ml log10_var vs log10_mean, after size factor')
        
        fig_var_mean(M,title='dd log10_var vs log10_mean, after size factor')
        
        fig_M2M1sqr(M,title='dd log10_M2 vs log10_mean^2, after size factor')

    return PC,PC_ml,gene_name

"""
    estimate the zero probability
"""
def dd_zero_anndata(data,verbose=True):
    if verbose: 
        start_time=time.time()
        print('# time start: 0.0s')
        
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    ## a new implementation without using the sparse matrix structure 
    Y = data.X    ## csr matrix
    A = Y.data
    JA = Y.indices
    gene_name = list(data.var_names)    
    w = zero_component_estimator(20)
    
    p0_ml = np.zeros([G],dtype=float)
    p0_dd = np.zeros([G],dtype=float)
    
    A_dd = np.zeros([A.shape[0]],dtype=float)
    for i,val in enumerate(w):
        A_dd[A==i] = val    
    
    temp_J_list = np.bincount(JA)
    temp_w_list = np.bincount(JA,weights=A_dd)        
    for i_gene in range(temp_J_list.shape[0]):       
        p0_ml[i_gene] += temp_J_list[i_gene] 
        p0_dd[i_gene] += temp_w_list[i_gene]
        
    p0_ml = 1-p0_ml/Nc
    p0_dd = p0_ml + p0_dd/Nc
        
    if verbose:   
        print('# total time: %0.1fs'%(time.time()-start_time))               
    return p0_dd,p0_ml,gene_name

## old implementation: 
    #Y = data.X    ## csr matrix
    #Y_data = Y.data
    #gene_name = list(data.var_names)    
    #w = zero_component_estimator(20)
    #
    ### first calculate the ml estimate
    #Y_data_ml = np.ones([Y_data.shape[0]],dtype=float)    ## keep track of the non-zero proportion 
    #Y_ml = sp.sparse.csr_matrix((Y_data_ml,Y.indices,Y.indptr))
    #p0_ml = 1-Y_ml.mean(axis=0) 
    #
    #
    ### maintain two copies of the csr matrix 
    #Y_data_dd = np.zeros([Y_data.shape[0]],dtype=float)
    #for i,val in enumerate(w):
    #    Y_data_dd[Y_data==i] = val  
    #Y_dd = sp.sparse.csr_matrix((Y_data_dd,Y.indices,Y.indptr))
    #p0_dd = p0_ml+Y_dd.mean(axis=0) 
    #
    #p0_ml = np.array(p0_ml).reshape(-1)
    #p0_dd = np.array(p0_dd).reshape(-1)

"""
    estimate the pairwise zero probability
"""
def dd_pairwise_zero_anndata(data,verbose=True):
    
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    Y = data.X    ## csr matrix
    A = Y.data
    IA = Y.indptr
    JA = Y.indices
    gene_name = list(data.var_names)    
    w = zero_component_estimator(20)
    
    ## fixit: remove the large entries     
    
    ## Maintain weights for the dd estimator. The weights for ml is always 1, no need of maintenance.
    A_dd = np.zeros([A.shape[0]],dtype=float)
    for i,val in enumerate(w):
        A_dd[A==i] = val      
    
    zero_matrix_dd = np.zeros([G,G],dtype=float)
    zero_matrix_ml = np.zeros([G,G],dtype=float)
    
    #check_time = time.time()
        
    temp_J_list = np.bincount(JA)
    temp_w_list = np.bincount(JA,weights=A_dd)        
    for i_gene in range(temp_J_list.shape[0]):       
        zero_matrix_ml[i_gene,:] += temp_J_list[i_gene] 
        zero_matrix_ml[:,i_gene] += temp_J_list[i_gene]
        zero_matrix_dd[i_gene,:] += temp_w_list[i_gene]
        zero_matrix_dd[:,i_gene] += temp_w_list[i_gene]
    
    ## a boardcast version of implementation which is unfortunately slower
    #zero_matrix_ml = zero_matrix_ml + temp_J_list
    #zero_matrix_ml = (zero_matrix_ml.T + temp_J_list).T
    #zero_matrix_dd = zero_matrix_dd + temp_w_list
    #zero_matrix_dd = (zero_matrix_dd.T + temp_w_list).T
        
    ## update the intersection part of the ml matrix
    temp_ml = sp.sparse.csr_matrix((np.ones([A.shape[0]],dtype=int),JA,IA))
    zero_matrix_ml = zero_matrix_ml - np.array((temp_ml.transpose().dot(temp_ml)).todense())
    
    ## update the intersection part of the dd matrix
    temp_dd = sp.sparse.csr_matrix((A_dd,JA,IA))
    zero_matrix_dd = zero_matrix_dd + np.array((temp_dd.transpose().dot(temp_dd)).todense())
    temp = np.array((temp_ml.transpose().dot(temp_dd)).todense())    
    zero_matrix_dd = zero_matrix_dd - temp - temp.T                
    #print('# check time: %0.3fs'%(time.time()-check_time))
    
    zero_matrix_ml = 1-zero_matrix_ml/Nc
    zero_matrix_dd = zero_matrix_dd/Nc + zero_matrix_ml
    
    if verbose:   
        print('# total time: %0.3fs'%(time.time()-start_time)) 
    return zero_matrix_dd,zero_matrix_ml,gene_name


## record: original implementation, slow
    #zero_matrix_dd = np.zeros([G,G],dtype=float)
    #zero_matrix_ml = np.zeros([G,G],dtype=float)
    #
    #for i_row in range(Nc):     
    #    column_index_in_row_i = JA[IA[i_row]:IA[i_row+1]]
    #    n_idx_in_row_i = len(column_index_in_row_i)
    #    A_dd_row_i = A_dd[IA[i_row]:IA[i_row+1]]   
    #    
    #    for i_col,val_col in enumerate(column_index_in_row_i):
    #        zero_matrix_ml[val_col,:] += 1 
    #        zero_matrix_ml[:,val_col] += 1
    #        zero_matrix_dd[val_col,:] += A_dd_row_i[i_col]
    #        zero_matrix_dd[:,val_col] += A_dd_row_i[i_col]
    #    
    #    for i_col in range(n_idx_in_row_i):
    #        for j_col in range(i_col,n_idx_in_row_i):
    #            zero_matrix_ml[column_index_in_row_i[i_col],column_index_in_row_i[j_col]] -=1
    #            zero_matrix_dd[column_index_in_row_i[i_col],column_index_in_row_i[j_col]] += \
    #                A_dd_row_i[i_col] * A_dd_row_i[j_col] - A_dd_row_i[i_col] - A_dd_row_i[j_col] 
    

"""
    calculate the weight for the zero component estimator 
"""

def zero_component_estimator(L,estimator_type='ET',param={'t':2},require_param=False):
    w = np.zeros([L+1],dtype=float)
    v_L = np.arange(L+1)    ## a length-L idx vector 
    
    if estimator_type == 'ML':
        w[0] = 1
        param_str = ''
    elif estimator_type == 'GT':
        t = param['t']
        w = (-t)**v_L
        param_str = 't=%0.2f'%t
    elif estimator_type == 'ET':
        t = param['t']
        n = 100
        k = np.ceil(0.5 * np.log2(n*t**2/(t-1)))
        w = (-t)**v_L * (1-sp.stats.binom.cdf(v_L-1,k,1/(t+1)))
        param_str = 't=%0.2f, k=%d, q=%0.2f'%(t,k,1/(t+1))
    elif estimator_type == 'SGT_bin':
        t = param['t']
        n = 100
        k = np.ceil(0.5 * np.log2(n*t**2/(t-1)) / np.log2(3))
        w = (-t)**v_L * (1-sp.stats.binom.cdf(v_L-1,k,2/(t+2)))
        param_str = 't=%0.2f, k=%d, q=%0.2f'%(t,k,2/(t+2))
    elif estimator_type == 'SGT_poi':
        t = param['t']
        n = 100
        r = 1 / (2*t) * np.log(n*(t+1)**2/(t-1))
        w = (-t)**v_L * (1-sp.stats.poisson.cdf(v_L-1,r))
        param_str = 't=%0.2f, r=%0.2f'%(t,r)
    else:
        print('## estimator_type not valid ##')
        return   
    
    #print(estimator_type)
    #print(w)
    if require_param: return w,param_str
    return w

"""
    visualize the weight for different zero component estimators
"""
def visualize_zero_component_estimator():
    v_lambda = np.linspace(0,10,101)
    n_lambda = v_lambda.shape[0]
    v_obs = np.arange(20)
    n_obs = v_obs.shape[0]
    P = np.zeros([n_lambda,n_obs],dtype=float)
    for i_lambda in range(n_lambda):
        for i_obs in range(n_obs):
            P[i_lambda,i_obs] = sp.stats.poisson.pmf(v_obs[i_obs],v_lambda[i_lambda])
    estimator_type_list = ['ML','GT','ET','SGT_bin','SGT_poi']
    estimator_param_list = [{},{'t':1},{'t':3},{'t':3},{'t':3}]
    element_weight={}
    estimator_param = {}
          
    for i_type,estimator_type in enumerate(estimator_type_list):
        w,param_str = zero_component_estimator\
                    (n_obs-1,estimator_type=estimator_type,\
                     param=estimator_param_list[i_type],\
                     require_param=True)
        element_weight[estimator_type] = P.dot(w)
        estimator_param[estimator_type] = param_str
        
    plt.figure(figsize=[18,5])
    for estimator_type in estimator_type_list:
        plt.plot(v_lambda,element_weight[estimator_type],label=estimator_type+' '+estimator_param[estimator_type])
    plt.legend()
    plt.show()


"""
    calculate some basic distributional quantities 
"""
def basic_cal(M):
    mean_ = M[:,0]
    var_ = (M[:,1]-M[:,0]**2).clip(min=0)
    cv_ = np.sqrt(var_) / mean_
    fano_ = var_/mean_
    return mean_,var_,cv_,fano_

def plot_xx(x1,x2,alpha=0.3,s=None,color='orange',xlabel=None,ylabel=None,lim=None):
    
    if lim is None:
        x_max = np.percentile(x1,99.9)+0.5
        x_min = np.percentile(x1,0.01)-0.5
        y_max = np.percentile(x2,99.9)+0.5
        y_min = np.percentile(x2,0.01)-0.5
    else:
        x_min,x_max,y_min,y_max = lim
        
    #x_max = max(np.percentile(x1,99.9),np.percentile(x2,99.9))+0.5
    #x_min = min(np.percentile(x1,0.1),np.percentile(x2,0.1))-0.5
    plt.scatter(x1,x2,alpha=alpha,color=color,s=s)
    #plt.scatter(x1,x2,color=color,alpha=alpha,s=20)
    plt.plot([x_min,x_max],[x_min,x_max],color='r')
    plt.xlim([x_min,x_max])    
    plt.ylim([y_min,y_max])
    if xlabel is not None: plt.xlabel(xlabel)    
    if ylabel is not None: plt.ylabel(ylabel)
        
def fig_var_mean(M,title=''):
    mean_,var_,_,_ = basic_cal(M)
    plt.figure(figsize=[18,5])
    plot_xx(np.log10(mean_.clip(min=1e-4)),np.log10(var_.clip(min=1e-4)))
    plt.title(title)
    plt.show()
    
def fig_M2M1sqr(M,title=''):
    M1,M2 = M[:,0],M[:,1]
    plt.figure(figsize=[18,5])
    plot_xx(2*np.log10(M1.clip(min=1e-4)),np.log10(M2.clip(min=1e-4)))
    plt.title(title)
    plt.show()
        
#def fig_xx_M2n(M1,M_ml1,M2,M_ml2,gene_list1,gene_list2,data_name,mean_fil=0.1):
#    name1,name2 = data_name
#    ## basic statistics 
#    mean1,var1,cv1,fano1 = basic_cal(M1)
#    mean_ml1,var_ml1,cv_ml1,fano_ml1 = basic_cal(M_ml1)
#    mean2,var2,cv2,fano2 = basic_cal(M2)
#    mean_ml2,var_ml2,cv_ml2,fano_ml2 = basic_cal(M_ml2)  
#    
#    cv1 = np.sqrt(M1[:,1])/M1[:,0]
#    cv_ml1 = np.sqrt(M_ml1[:,1])/M_ml1[:,0]
#    cv2 = np.sqrt(M2[:,1])/M2[:,0]
#    cv_ml2 = np.sqrt(M_ml2[:,1])/M_ml2[:,0]
#    
#    ## consistency     
#    common_gene_ = set(gene_list1) & set(gene_list2)
#    idx1 = []
#    idx2 = []
#    common_gene = []
#    n_gene=0
#    for idx,gene in enumerate(common_gene_):
#        if mean1[gene_list1.index(gene)]<mean_fil or mean2[gene_list2.index(gene)]<mean_fil: continue           
#        if cv1[gene_list1.index(gene)]==0 or cv2[gene_list2.index(gene)]==0: continue           
#        idx1.append(gene_list1.index(gene))
#        idx2.append(gene_list2.index(gene))
#        n_gene+=1
#        common_gene.append(gene)
#    
#    
#    x_max = max(np.percentile(np.log10(cv_ml1[idx1].clip(min=1e-2)),99.9),np.percentile(np.log10(cv1[idx1].clip(min=1e-2)),99.9))+0.5
#    x_min = min(np.percentile(np.log10(cv_ml1[idx1].clip(min=1e-2)),0.01),np.percentile(np.log10(cv1[idx1].clip(min=1e-2)),0.01))-0.5
#    y_max = max(np.percentile(np.log10(cv_ml2[idx2].clip(min=1e-2)),99.9),np.percentile(np.log10(cv2[idx2].clip(min=1e-2)),99.9))+0.5
#    y_min = min(np.percentile(np.log10(cv_ml2[idx2].clip(min=1e-2)),0.01),np.percentile(np.log10(cv2[idx2].clip(min=1e-2)),0.01))-0.5
#    lim = [x_min,x_max,y_min,y_max]
#    
#    print('ml: above=%d, below=%d, avg_r log10(cv2/cv1)=%0.3f'
#          %(np.sum(cv_ml1[idx1]<cv_ml2[idx2]),
#            np.sum(cv_ml1[idx1]>cv_ml2[idx2]),
#            (np.log10(cv_ml2[idx2].clip(min=1e-2))-np.log10(cv_ml1[idx1].clip(min=1e-2))).mean()))
#    print('dd: above=%d, below=%d, avg_r log10(cv2/cv1)=%0.3f'
#          %(np.sum(cv1[idx1]<cv2[idx2]),
#            np.sum(cv1[idx1]>cv2[idx2]),
#            (np.log10(cv2[idx2].clip(min=1e-2))-np.log10(cv1[idx1].clip(min=1e-2))).mean()))
#    
#    print(cv1[idx1].min(),cv2[idx2].min())
#    print(len(idx2),len(common_gene))
#    temp = np.where(cv2[idx2] == cv2[idx2].min())[0][0]
#    print(common_gene[temp])
#    
#    plt.figure(figsize=[12,12])
#    plt.subplot(221)
#    plot_xx(np.log10(cv_ml1[idx1].clip(min=1e-2)),np.log10(cv_ml2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,color='navy',lim=lim,s=20,alpha=0.5)
#    plt.title('cv_ml')
#    plt.subplot(222)
#    plot_xx(np.log10(cv1[idx1].clip(min=1e-2)),np.log10(cv2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,lim=lim,s=20,alpha=0.5)
#    plt.title('cv_dd')
#    plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
#    plt.show()
        
def fig_xx_cv(M1,M_ml1,M2,M_ml2,gene_list1,gene_list2,data_name,mean_fil=0.1,s=20,margin=0.5):
    name1,name2 = data_name
    ## basic statistics 
    mean1,var1,cv1,fano1 = basic_cal(M1)
    mean_ml1,var_ml1,cv_ml1,fano_ml1 = basic_cal(M_ml1)
    mean2,var2,cv2,fano2 = basic_cal(M2)
    mean_ml2,var_ml2,cv_ml2,fano_ml2 = basic_cal(M_ml2)    
    
    ## consistency     
    common_gene_ = set(gene_list1) & set(gene_list2)
    idx1 = []
    idx2 = []
    common_gene = []
    n_gene=0
    for idx,gene in enumerate(common_gene_):
        if mean1[gene_list1.index(gene)]<mean_fil or mean2[gene_list2.index(gene)]<mean_fil: continue           
        if cv1[gene_list1.index(gene)]==0 or cv2[gene_list2.index(gene)]==0: continue           
        #if cv1[gene_list1.index(gene)]<0.1 or cv2[gene_list2.index(gene)]<0.1: continue           
        idx1.append(gene_list1.index(gene))
        idx2.append(gene_list2.index(gene))
        n_gene+=1
        common_gene.append(gene)
    
    
    x_max = max(np.percentile(np.log10(cv_ml1[idx1].clip(min=1e-2)),99.9),np.percentile(np.log10(cv1[idx1].clip(min=1e-2)),99.9))+margin
    x_min = min(np.percentile(np.log10(cv_ml1[idx1].clip(min=1e-2)),0.01),np.percentile(np.log10(cv1[idx1].clip(min=1e-2)),0.01))-margin
    y_max = max(np.percentile(np.log10(cv_ml2[idx2].clip(min=1e-2)),99.9),np.percentile(np.log10(cv2[idx2].clip(min=1e-2)),99.9))+margin
    y_min = min(np.percentile(np.log10(cv_ml2[idx2].clip(min=1e-2)),0.01),np.percentile(np.log10(cv2[idx2].clip(min=1e-2)),0.01))-margin
    lim = [x_min,x_max,y_min,y_max]
    
    print('ml: above=%d, below=%d, avg_r log10(cv2/cv1)=%0.3f'
          %(np.sum(cv_ml1[idx1]<cv_ml2[idx2]),
            np.sum(cv_ml1[idx1]>cv_ml2[idx2]),
            (np.log10(cv_ml2[idx2].clip(min=1e-2))-np.log10(cv_ml1[idx1].clip(min=1e-2))).mean()))
    print('dd: above=%d, below=%d, avg_r log10(cv2/cv1)=%0.3f'
          %(np.sum(cv1[idx1]<cv2[idx2]),
            np.sum(cv1[idx1]>cv2[idx2]),
            (np.log10(cv2[idx2].clip(min=1e-2))-np.log10(cv1[idx1].clip(min=1e-2))).mean()))
    
    plt.figure(figsize=[12,12])
    plt.subplot(221)
    plot_xx(np.log10(cv_ml1[idx1].clip(min=1e-2)),np.log10(cv_ml2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,color='navy',lim=lim,s=s,alpha=0.5)
    plt.title('cv_ml')
    plt.subplot(222)
    plot_xx(np.log10(cv1[idx1].clip(min=1e-2)),np.log10(cv2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,lim=lim,s=s,alpha=0.5)
    plt.title('cv_dd')
    plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
    plt.show()
    
def fig_xx_zero(p0_dd1,p0_ml1,p0_dd2,p0_ml2,gene_list1,gene_list2,data_name,mean_fil=0.1,s=20,margin=0.5):
    name1,name2 = data_name
    
    ## consistency     
    common_gene_ = set(gene_list1) & set(gene_list2)
    idx1 = []
    idx2 = []
    common_gene = []
    n_gene=0
    for idx,gene in enumerate(common_gene_):
        idx1.append(gene_list1.index(gene))
        idx2.append(gene_list2.index(gene))
        n_gene+=1
        common_gene.append(gene)
    
    
    x_max = max(np.percentile(np.log10(p0_ml1[idx1].clip(min=1e-2)),99.9),np.percentile(np.log10(p0_dd1[idx1].clip(min=1e-2)),99.9))+margin
    x_min = min(np.percentile(np.log10(p0_ml1[idx1].clip(min=1e-2)),0.01),np.percentile(np.log10(p0_dd1[idx1].clip(min=1e-2)),0.01))-margin
    y_max = max(np.percentile(np.log10(p0_ml2[idx2].clip(min=1e-2)),99.9),np.percentile(np.log10(p0_dd2[idx2].clip(min=1e-2)),99.9))+margin
    y_min = min(np.percentile(np.log10(p0_ml2[idx2].clip(min=1e-2)),0.01),np.percentile(np.log10(p0_dd2[idx2].clip(min=1e-2)),0.01))-margin
    lim = [x_min,x_max,y_min,y_max]
    
    print('ml: above=%d, below=%d, avg_r log10(p0_2/p0_1)=%0.3f'
          %(np.sum(p0_ml1[idx1]<p0_ml2[idx2]),
            np.sum(p0_ml1[idx1]>p0_ml2[idx2]),
            (np.log10(p0_ml2[idx2].clip(min=1e-2))-np.log10(p0_ml1[idx1].clip(min=1e-2))).mean()))
    print('dd: above=%d, below=%d, avg_r log10(p0_2/p0_1)=%0.3f'
          %(np.sum(p0_dd1[idx1]<p0_dd2[idx2]),
            np.sum(p0_dd1[idx1]>p0_dd2[idx2]),
            (np.log10(p0_dd2[idx2].clip(min=1e-2))-np.log10(p0_dd1[idx1].clip(min=1e-2))).mean()))
    
    plt.figure(figsize=[12,12])
    plt.subplot(221)
    plot_xx(np.log10(p0_ml1[idx1].clip(min=1e-2)),np.log10(p0_ml2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,color='navy',lim=lim,s=s,alpha=0.5)
    plt.title('p0_ml')
    plt.subplot(222)
    plot_xx(np.log10(p0_dd1[idx1].clip(min=1e-2)),np.log10(p0_dd2[idx2].clip(min=1e-2)),xlabel=name1,ylabel=name2,lim=lim,s=s,alpha=0.5)
    plt.title('p0_dd')
    #plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
    plt.show()

def v_score(mean_,cv_):
    log_cv = np.log(cv_)
    log_mean = np.log(mean_).reshape([mean_.shape[0],1])
    LR = LinearRegression()
    LR.fit(log_mean,log_cv)
    fit_cv = LR.predict(log_mean)    
    score = fit_cv-log_cv
    plt.figure(figsize=[12,5])
    plt.scatter(log_mean,log_cv,alpha=0.1)   
    plt.scatter(log_mean,fit_cv)
    plt.show()
    
    return score

def fig_xx_rank(M1,M_ml1,M2,M_ml2,gene_list1,gene_list2,data_name,mean_fil=0.1):
    
    name1,name2 = data_name
    ## basic statistics 
    mean1,var1,cv1,fano1 = basic_cal(M1)
    mean_ml1,var_ml1,cv_ml1,fano_ml1 = basic_cal(M_ml1)
    mean2,var2,cv2,fano2 = basic_cal(M2)
    mean_ml2,var_ml2,cv_ml2,fano_ml2 = basic_cal(M_ml2)    

    ## consistency     
    common_gene_ = set(gene_list1) & set(gene_list2)
    idx1 = []
    idx2 = []
    common_gene = []
    n_gene=0
    for idx,gene in enumerate(common_gene_):
        if mean1[gene_list1.index(gene)]<mean_fil or mean2[gene_list2.index(gene)]<mean_fil: continue           
        if cv1[gene_list1.index(gene)]==0 or cv2[gene_list2.index(gene)]==0: continue           
        idx1.append(gene_list1.index(gene))
        idx2.append(gene_list2.index(gene))
        n_gene+=1
        common_gene.append(gene)
    idx1 = np.array(idx1,dtype=int)
    idx2 = np.array(idx2,dtype=int)
    
    
    
    ## ranking 
    #score1 = v_score(mean1[idx1],cv1[idx1])
    #score2 = v_score(mean2[idx2],cv2[idx2])
    #score_ml1 = v_score(mean_ml1[idx1],cv_ml1[idx1])
    #score_ml2 = v_score(mean_ml2[idx2],cv_ml2[idx2])
    
    #score1 = v_score(mean1[idx1],fano1[idx1])
    #score2 = v_score(mean2[idx2],fano2[idx2])
    #score_ml1 = v_score(mean_ml1[idx1],fano_ml1[idx1])
    #score_ml2 = v_score(mean_ml2[idx2],fano_ml2[idx2])
    
    #rank_cv1 = rank(score1,opt='descent')
    #rank_cv2 = rank(score2,opt='descent')
    #rank_cv_ml1 = rank(score_ml1,opt='descent')
    #rank_cv_ml2 = rank(score_ml2,opt='descent')
    #
    #plt.figure(figsize=[12,12])
    #plt.subplot(221)
    #plot_xx(score_ml1,score_ml2,xlabel=name1,ylabel=name2,color='navy')
    #plt.title('cv_ml')
    #plt.subplot(222)
    #plot_xx(score1,score2,xlabel=name1,ylabel=name2)
    #plt.title('cv_dd')
    #plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
    #plt.show()
    
    
    rank_cv1 = rank(cv1[idx1],opt='descent')
    rank_cv2 = rank(cv2[idx2],opt='descent')
    rank_cv_ml1 = rank(cv_ml1[idx1],opt='descent')
    rank_cv_ml2 = rank(cv_ml2[idx2],opt='descent')
    
    
    plt.figure(figsize=[12,12])
    plt.subplot(221)
    plot_xx(rank_cv_ml1,rank_cv_ml2,color='navy',alpha=0.1,xlabel=name1,ylabel=name2)
    plt.title('rank_cv_ml')
    plt.subplot(222)
    plot_xx(rank_cv1,rank_cv2,color='orange',alpha=0.1,xlabel=name1,ylabel=name2)
    plt.title('rank_cv_dd')
    plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
    plt.show()

    topk = np.arange(10,min(1000,rank_cv1.shape[0]),10) 
    overlap=np.zeros([topk.shape[0]])
    overlap_ml = np.zeros([topk.shape[0]])
    for i,ik in enumerate(topk):
        overlap[i]=((rank_cv1<ik)&(rank_cv2<ik)).sum()
        overlap_ml[i] = ((rank_cv_ml1<ik)&(rank_cv_ml2<ik)).sum()

    plt.figure(figsize=[12,9])
    plt.plot(topk,overlap_ml,marker='o',color='navy',label='ml')
    plt.plot(topk,overlap,marker='o',color='orange',label='dd')
    plt.legend()
    plt.suptitle('mean_fil=%0.2f, n_gene=%d'%(mean_fil,n_gene))
    plt.show()
    