""" 
    scdd: "Single Cell Density Deconvolution"
    a self-contained module for moment estimation only
    some nice figure generation functions are included
    the data are assumed to be stored with an anndata format    
"""

import numpy as np
import scipy as sp
import pandas as pd
import itertools
from scipy import stats
from scipy import special
import matplotlib.pyplot as plt
import time
import seaborn as sns
from multiprocessing import Pool
import time
from sklearn.linear_model import LinearRegression
import scanpy.api as sc
import logging
import os

from sceb.b_spline_nd import *
from sceb.util import *

"""
    Print the size information of the dataset
"""
def get_info(data,logger=None):
    Nc,G = data.shape
    Nr = data.X.sum()/Nc
    Nr_bar = Nr/G
    if logger is None:
        print('## Nc=%d, G=%d, Nr=%0.2f, Nr_bar=%0.2f'%(Nc,G,Nr,Nr_bar))  
    else:
        logger.info('## Nc=%d, G=%d, Nr=%0.2f, Nr_bar=%0.2f'%(Nc,G,Nr,Nr_bar))  
    return Nc,G,Nr,Nr_bar

""" 
    calculate the size factor
"""
def dd_size_factor(data,verbose=False):
    X=data.X
    Nrc = np.array(X.sum(axis=1)).reshape(-1)
    Nr = Nrc.mean()
    size_factor = Nrc/Nr
    
    if verbose: 
        print('Nr=%d'%Nr)
        print('size factor',np.percentile(size_factor,[0,0.01,0.1,99,99.9,100]))
        plt.figure(figsize=[12,5])
        plt.hist(size_factor,bins=20)
        plt.show()
    return size_factor

"""
    the subsample function for anndata input
"""
def subsample_anndata(adata,Nr_new,Nc_new,random_state=0,verbose=True):
    np.random.seed(random_state)
    
    adata = adata.copy()
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    
    ## sub-sample the cells
    Nc,G=adata.shape
    Nr = adata.X.sum()/Nc
    if verbose: print('before cell subsamp',adata.shape)
    sc.pp.subsample(adata,Nc_new/Nc,random_state)
    Nc,G=adata.shape
    if verbose: 
        print('after cell subsamp',adata.shape)
        print('#time sub-sample cells: %0.4fs'%(time.time()-start_time)) 
        
        
    
    ## sub-sample the counts for each cell        
    #Nr = adata.X.sum()/Nc
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
    adata.X = X_new
        
    if verbose: print('#time sub-sample counts: %0.4fs\n'%(time.time()-start_time)) 
    return adata
    
""" 
    calculate the moments using 1. ml, 2. dd
"""

def dd_1d_moment(data,size_factor=None,verbose=True,k=2,Nr=1):
    if k>4:
        print('### The program only outputs at most 4 moments')    
    
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    #Nr = data.X.sum()/Nc   
    #Nr = 1 # fixit
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    X = data.X ## csr file   
    M_ml = np.zeros([k,G],dtype=float)
    M_dd = np.zeros([k,G],dtype=float)
    
    if size_factor is None: 
        row_weight = np.ones([Nc])
    else:
        row_weight = 1/size_factor
        
    ## moment calculation
    if 1<=k:
        M1_ = sp.sparse.csc_matrix.dot(row_weight.reshape([1,-1]),X)
        M1_ = np.array(M1_).reshape(-1)/Nc
        M_ml[0] = M1_
        M_dd[0] = M1_
    
    if 2<=k:
        M1_ = sp.sparse.csc_matrix.dot((row_weight**2).reshape([1,-1]),X)
        M1_ = np.array(M1_).reshape(-1)/Nc        
        M2_ = sp.sparse.csc_matrix.dot((row_weight**2).reshape([1,-1]),X.power(2))
        M2_ = np.array(M2_).reshape(-1)/Nc
        
        M_ml[1] = M2_
        M_dd[1] = M2_ - M1_
        
    
    if 3<=k:
        M1_ = sp.sparse.csc_matrix.dot((row_weight**3).reshape([1,-1]),X)
        M1_ = np.array(M1_).reshape(-1)/Nc        
        M2_ = sp.sparse.csc_matrix.dot((row_weight**3).reshape([1,-1]),X.power(2))
        M2_ = np.array(M2_).reshape(-1)/Nc
        M3_ = sp.sparse.csc_matrix.dot((row_weight**3).reshape([1,-1]),X.power(3))
        M3_ = np.array(M3_).reshape(-1)/Nc
        
        M_ml[2] = M3_
        M_dd[2] = M3_ - 3*M2_ + 2*M1_
    
    if 4<=k:
        M1_ = sp.sparse.csc_matrix.dot((row_weight**4).reshape([1,-1]),X)
        M1_ = np.array(M1_).reshape(-1)/Nc        
        M2_ = sp.sparse.csc_matrix.dot((row_weight**4).reshape([1,-1]),X.power(2))
        M2_ = np.array(M2_).reshape(-1)/Nc
        M3_ = sp.sparse.csc_matrix.dot((row_weight**4).reshape([1,-1]),X.power(3))
        M3_ = np.array(M3_).reshape(-1)/Nc
        M4_ = sp.sparse.csc_matrix.dot((row_weight**4).reshape([1,-1]),X.power(4))
        M4_ = np.array(M4_).reshape(-1)/Nc
        
        M_ml[3] = M4_
        M_dd[3] = M4_ - 6*M3_ + 11*M2_ - 6*M1_    
    
    ## normalize by Nr
    for i in range(k):
        M_ml[i] = M_ml[i]/Nr**(i+1)
        M_dd[i] = M_dd[i]/Nr**(i+1)
            
    if verbose: 
        print('#total: %0.2fs'%(time.time()-start_time))
    return M_ml,M_dd

""" 
    Calculate the covariance matrix as well as the 
    PC (Pearson correlation) using ml and dd
"""

def dd_covariance(data,size_factor=None,verbose=True,return_ml=False,PC_prune=True):           
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    Nr = data.X.sum()/Nc    
    if verbose: 
        print('n_cell=%d, n_gene=%d, Nr=%0.1f'%(Nc,G,Nr))
    
    X = data.X ## csr file
    gene_name = list(data.var_names)
    
    ## normalize by the size factor    
    if size_factor is not None: 
        row_weight = 1/size_factor
        X = assign_row_weight(X,row_weight)
        
    ## Statistics after the first normalization 
    mean_dd = np.array(X.mean(axis=0)).reshape(-1)    
    M2_dd = np.array((X.transpose().dot(X)/Nc).todense())  
    
    ## Double normalization to get what we need
    if size_factor is not None: 
        X = assign_row_weight(X,row_weight)
    
    ## Statistics after the second normalization
    mean_doublenorm_dd = np.array(X.mean(axis=0)).reshape(-1)
    
    ## Bias correction
    M2_dd -= np.diag(mean_doublenorm_dd)
    
    ## Covariance matrix 
    temp = mean_dd.reshape([G,1])
    cov_dd = M2_dd - temp.dot(temp.T)
    
    ## clip the variance estimation to be 0: ???
    diag_cov_dd = np.diag(cov_dd)
    
    ## bad indeces
    index_bad = np.zeros([G],dtype=bool)
    index_bad[diag_cov_dd<=1e-1/9] = True
    index_bad[(diag_cov_dd/mean_dd)<0.1] = True
    
    np.fill_diagonal(cov_dd,diag_cov_dd.clip(min=1e-12))
        
    ## Pearson correlation
    std_dd = np.sqrt(diag_cov_dd)    
    #std_dd = np.sqrt(diag_cov_dd)
    std_dd = std_dd.reshape([G,1])
    PC_dd = cov_dd/(std_dd.dot(std_dd.T))
    PC_dd = PC_dd.clip(min=-1,max=1)
    
    ## for the bad index, remove the estimation: ???
    if PC_prune:
        PC_dd[:,index_bad] = 0
        PC_dd[index_bad,:] = 0
    
    if verbose: 
        print('#total: %0.2fs'%(time.time()-start_time))
    return mean_dd,cov_dd,PC_dd


# fixit: change it back to matrix multiplication
def assign_row_weight(X,row_weight): 
    X = X.astype(np.float64)
    X = (X.T.multiply(row_weight)).T.tocsr()
    return X

def assign_row_weight_with_copy(X_,row_weight): 
    X = X_.copy()
    X = X.astype(np.float64)
    X = (X.T.multiply(row_weight)).T.tocsr()
    return X

def ml_covariance(data,size_factor=None,verbose=True):
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    Nr = data.X.sum()/Nc    
    if verbose: 
        print('n_cell=%d, n_gene=%d, Nr=%0.1f'%(Nc,G,Nr))
    
    X = data.X ## csr file
    gene_name = list(data.var_names)
    
    ## normalize by the size factor    
    if size_factor is not None: 
        X = assign_row_weight(X,1/size_factor)
        
    ## Mean 
    mean_ml = np.array(X.mean(axis=0)).reshape(-1)
    
    ## Second moment
    M2_ml = np.array((X.transpose().dot(X)/Nc).todense())  
    
    ## Covariance matrix
    temp = mean_ml.reshape([G,1])
    cov_ml = M2_ml - temp.dot(temp.T)
    
    ## Pearson correlation
    std_ml = np.diag(cov_ml).clip(min=0)
    std_ml = np.sqrt(std_ml)    
    std_ml = std_ml.reshape([G,1])
    PC_ml = cov_ml/(std_ml.dot(std_ml.T))
    PC_ml = PC_ml.clip(min=-1,max=1)
    
    ## normalize by Nr:
    #mean_ml /= Nr
    #cov_ml /= Nr**2
    
    if verbose: 
        print('#total: %0.2fs'%(time.time()-start_time))
    
    return mean_ml,np.array(cov_ml),np.array(PC_ml)  

"""
    estimate the zero probability
"""
def dd_zero_prob(data,verbose=True):
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
    return p0_ml,p0_dd

"""
    Estimate the inactive probability
    relative_depth = Nr / Nr0, where Nr0 is an anchor dataset
"""
def dd_inactive_prob(data,relative_depth=1,size_factor=None,verbose=True):
    def sub_dd_inactive_prob(Y_sub,t_sub,Nc_sub,n=500):
        A = Y_sub.data
        JA = Y_sub.indices
        if t_sub<1: 
            L = np.log(0.005)/np.log(1-t_sub)
            #L = np.log(0.01)/np.log(1-t_sub)
        else:
            L=10
        w = smooth_zero_estimator(L,t=t_sub,n=n,require_param=False)
        #print(w)
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
        p0_ml = 1-p0_ml/Nc_sub
        p0_dd = p0_ml + p0_dd/Nc_sub        
        return p0_ml,p0_dd
        
    if verbose: 
        start_time=time.time()
        print('# time start: 0.0s')
        
    Nc,G = data.shape
    n = np.min([500,Nc])
    Y = data.X
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    if size_factor is None:
        p0_ml,p0_dd = sub_dd_inactive_prob(Y,1/relative_depth,Nc,n=n)
    else:
        tc = 1/relative_depth/size_factor
        amp = np.max([1/np.percentile(tc,0.1),20])
        tc = np.round(tc*amp)/amp
        tc = tc.clip(min=1/amp)
        p0_ml = np.zeros([G],dtype=float)
        p0_dd = np.zeros([G],dtype=float)
        for tc_ in np.unique(tc):
            Nc_sub = np.sum(tc==tc_)
            p0_ml_sub,p0_dd_sub = sub_dd_inactive_prob(Y[tc==tc_],tc_,Nc_sub,n=n)
            p0_ml += p0_ml_sub*Nc_sub/Nc
            p0_dd += p0_dd_sub*Nc_sub/Nc
        
    if verbose:   
        print('# total time: %0.1fs'%(time.time()-start_time))               
    return p0_ml.clip(min=0),p0_dd.clip(min=0)

"""
    estimate the pairwise zero probability
"""
def dd_pairwise_zero_prob(data,verbose=True):
    
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
    
    p0_ml,p0_dd = dd_zero_prob(data)
    np.fill_diagonal(zero_matrix_ml,p0_ml)
    np.fill_diagonal(zero_matrix_dd,p0_dd)
    
    if verbose:   
        print('# total time: %0.3fs'%(time.time()-start_time)) 
    return zero_matrix_ml,zero_matrix_dd



"""
    Estimate the inactive probability
    relative_depth = Nr / Nr0, where Nr0 is an anchor dataset
"""
def dd_pairwise_inactive_prob(data,relative_depth=1,size_factor=None,verbose=True):
    def sub_dd_pairwise_inactive_prob(Y_sub,t_sub,Nc_sub,n=500):
        A = Y_sub.data
        IA = Y_sub.indptr
        JA = Y_sub.indices
        
        if t_sub<1: 
            L = np.log(0.005)/np.log(1-t_sub)
        else:
            L=10
            
        w = smooth_zero_estimator(L,t=t_sub,n=n,require_param=False)

        ## Maintain weights for the dd estimator. The weights for ml is always 1, no need of maintenance.
        A_dd = np.zeros([A.shape[0]],dtype=float)
        for i,val in enumerate(w):
            A_dd[A==i] = val      

        zero_matrix_dd = np.zeros([G,G],dtype=float)
        zero_matrix_ml = np.zeros([G,G],dtype=float)

        temp_J_list = np.bincount(JA)
        temp_w_list = np.bincount(JA,weights=A_dd)        
        for i_gene in range(temp_J_list.shape[0]):       
            zero_matrix_ml[i_gene,:] += temp_J_list[i_gene] 
            zero_matrix_ml[:,i_gene] += temp_J_list[i_gene]
            zero_matrix_dd[i_gene,:] += temp_w_list[i_gene]
            zero_matrix_dd[:,i_gene] += temp_w_list[i_gene]
            
        ## update the intersection part of the ml matrix
        temp_ml = sp.sparse.csr_matrix((np.ones([A.shape[0]],dtype=int),JA,IA),shape=(Nc_sub,G))
        zero_matrix_ml = zero_matrix_ml - np.array((temp_ml.transpose().dot(temp_ml)).todense())

        ## update the intersection part of the dd matrix
        temp_dd = sp.sparse.csr_matrix((A_dd,JA,IA),shape=(Nc_sub,G))
        zero_matrix_dd = zero_matrix_dd + np.array((temp_dd.transpose().dot(temp_dd)).todense())
        temp = np.array((temp_ml.transpose().dot(temp_dd)).todense())    
        zero_matrix_dd = zero_matrix_dd - temp - temp.T                

        zero_matrix_ml = 1-zero_matrix_ml/Nc_sub
        zero_matrix_dd = zero_matrix_dd/Nc_sub + zero_matrix_ml
        return zero_matrix_ml,zero_matrix_dd
    
    if verbose: 
        start_time=time.time()
        print('# time start: 0.0s')
        
    Nc,G = data.shape
    n = np.min([Nc,500])
    Y = data.X
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    
    if size_factor is None:
        zero_matrix_ml,zero_matrix_dd = sub_dd_pairwise_inactive_prob(Y,1/relative_depth,Nc,n=n)
    else:
        tc = 1/relative_depth/size_factor
        #tc = np.round(tc*20)/20
        amp = np.max([1/np.percentile(tc,0.1),20])
        tc = np.round(tc*amp)/amp
        tc = tc.clip(min=1/amp)
        zero_matrix_ml = np.zeros([G,G],dtype=float)
        zero_matrix_dd = np.zeros([G,G],dtype=float)
        for tc_ in np.unique(tc):
            Nc_sub = np.sum(tc==tc_)
            zero_matrix_ml_sub,zero_matrix_dd_sub = sub_dd_pairwise_inactive_prob(Y[tc==tc_],tc_,Nc_sub,n=n)
            zero_matrix_ml += zero_matrix_ml_sub*Nc_sub/Nc
            zero_matrix_dd += zero_matrix_dd_sub*Nc_sub/Nc
    
    ## fill in the diagonal elements
    p0_ml,p0_dd = dd_inactive_prob(data,relative_depth=relative_depth,size_factor=size_factor,verbose=False)
    np.fill_diagonal(zero_matrix_ml,p0_ml)
    np.fill_diagonal(zero_matrix_dd,p0_dd)
    
    if verbose:   
        print('# total time: %0.1fs'%(time.time()-start_time))               
    return zero_matrix_ml,zero_matrix_dd

"""
    Calculate the MI matrix based on the zero probability estimation
"""
def zero_to_mi(p0_joint,up_reg_gene=False):
    #p0_joint = p0_joint.clip(min=1e-6,max=1-1e-6)
    bad_index_threshold = 0.01
    bad_index = np.diag(p0_joint)<bad_index_threshold
    bad_index[np.diag(1-p0_joint)<bad_index_threshold] = True

    precision = 1e-10
    G = p0_joint.shape[0]
    mi_matrix = np.zeros([G,G],dtype=float)
        
    p0_margin = np.diag(p0_joint)
    temp = p0_margin.reshape([-1,1])
    up_reg = p0_joint > temp.dot(temp.T)
    np.fill_diagonal(up_reg,0)
    
    ## Marginal entropy
    H_margin = -p0_margin*np.log(p0_margin.clip(min=precision)) - (1-p0_margin)*np.log((1-p0_margin).clip(min=precision))
    mi_matrix = mi_matrix + H_margin
    mi_matrix = (mi_matrix.T + H_margin).T
    
    ## Joint entropy
    mi_matrix += p0_joint*np.log(p0_joint.clip(min=precision)) # first term 
    temp = -p0_joint+p0_margin # second term 
    mi_matrix += temp*np.log(temp.clip(min=precision))
    temp = (-p0_joint.T+p0_margin).T # third term 
    mi_matrix += temp*np.log(temp.clip(min=precision))
    temp = 1 + p0_joint
    temp = temp - p0_margin
    temp = (temp.T - p0_margin).T
    mi_matrix += temp*np.log(temp.clip(min=precision))
    #np.fill_diagonal(mi_matrix,1)
    mi_matrix = mi_matrix.clip(min=precision)    
    
    mi_matrix[bad_index,:] = 0
    mi_matrix[:,bad_index] = 0
    mi_matrix[p0_joint<bad_index_threshold] = 0
    
    if up_reg_gene:
        return mi_matrix*up_reg
    else:
        return mi_matrix
    
"""
    calculate the weight for the zero component estimator 
"""

def zero_component_estimator(L,estimator_type='ET',param={'t':2},require_param=False):
    w = np.zeros([L+1],dtype=float)
    v_L = np.arange(L+1)    ## a length-L idx vector 
    
    if estimator_type == 'Plug-in':
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
    
    if require_param: return w,param_str
    return w

"""
    Estimate the smooth version of the zero probability
    t = kappa / Nr*gamma_c
""" 
def smooth_zero_estimator(L,t=1,n=500,require_param=False,restrict_t=True):
    #print('L',L)
    L = np.ceil(L)
    if restrict_t:
        t = min(t,5) # for robustness consideration
    v_L = np.arange(L+1)    ## a length-(L+1) idx vector 
    t_ = t-1    
    w = (-t_)**v_L
    
    if t>2:
        #n = 500
        k = np.ceil(0.5 * np.log2(n*t_**2/(t_-1)))
        w = w * (1-sp.stats.binom.cdf(v_L-1,k,1/(t_+1)))
    
        param_str = 't=%0.2f, k=%d, q=%0.2f'%(t,k,1/(t+1))  
    
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
    estimator_type_list = ['Plug-in','GT','ET','SGT_bin','SGT_poi']
    estimator_type_list = ['Plug-in','ET','ET','ET','ET']
    estimator_param_list = [{},{'t':1.1},{'t':1.5},{'t':2},{'t':3}]
    element_weight = []
    estimator_param = []
    
    for i_type,estimator_type in enumerate(estimator_type_list):
        w,param_str = zero_component_estimator\
                    (n_obs-1,estimator_type=estimator_type,\
                     param=estimator_param_list[i_type],\
                     require_param=True)
        param_str = estimator_type+' '+param_str
        element_weight.append(P.dot(w))
        estimator_param.append(param_str)
        
    plt.figure(figsize=[6,5])
    for i in range(len(element_weight)):
        plt.plot(v_lambda,element_weight[i],label=estimator_param[i])
    plt.legend()
    plt.show()
    
""" 
    1d distribution estimation (try the simpliest version, no fancy stuffs)
"""
def Y_gen(p,x,Nc=10000,Nr=5,verbose=False):
    np.random.seed(910624)
    n_supp = p.shape[0]
    x_samp = np.random.choice(x,Nc,p=p,replace=True)
    Y=np.random.poisson(x_samp*Nr)    
    return Y,x_samp

"""
    1d distribution estimation (try the simpliest version, no fancy stuffs)
    c_res: resolution parameter
    
    Regularization is not applied on the zero component
"""
## distribution estimation: density deconvolution
def dd_distribution(Y,gamma=None,c_reg=1e-4,n_degree=5,zero_inflate=True,verbose=False):#,zero_esti=False):   
    ## setting parameters     
    if gamma is None:
        gamma = cal_gamma(Y)
                
    if verbose: 
        print('n_degree: %s, c_reg: %s, gamma: %s\n'%(str(n_degree),str(c_reg),str(gamma)))
    
    ## Split the sample into high count and low count
    ## Then convert the counts into histogram
    cutoff = 20*(gamma<20)+int(gamma+np.sqrt(gamma))*(gamma>=20)
    Y_pdf_high,Y_supp_high = counts2pdf_1d(Y[Y>cutoff])    
    Y_pdf,Y_supp = counts2pdf_1d(Y[Y<=cutoff])   
    n_high = np.sum(Y>cutoff)
    n_low  = np.sum(Y<=cutoff)   
        
    ## Optimization
    x          = np.linspace(0,1,201)**2
    
    Q,n_degree = Q_gen_ND(x,n_degree=n_degree,zero_inflate=zero_inflate)        
    P_model = Pmodel_cal(x,Y_supp,gamma)
    
    alpha = np.ones([n_degree],dtype=float) * 0.1
    lr = 1
    l_old = -100
    l = -99
    i_itr = -1
    
    while np.absolute(l-l_old)>1e-8:
        i_itr += 1
        l_old = l
        grad = grad_cal(alpha,Y_pdf,P_model,Q,c_reg)
        alpha += lr*grad/np.linalg.norm(grad)
        
        l = l_cal(alpha,Y_pdf,P_model,Q,c_reg)
        if l<l_old:
            lr/=2
        ## adjust the regularization penalty to be 0.001 of the likelihood
        if i_itr%20==0 and i_itr!=0:
            #c_reg = np.absolute(l) / np.linalg.norm(alpha[1:]) * 1e-4           
            c_reg = np.absolute(l) / np.linalg.norm(alpha) * 1e-4           
    
    alpha_hat = alpha
    
    ## Post-processing the result
    p_hat     = px_cal(Q,alpha_hat) 
    
    #if zero_esti:
    #    p_hat = p_hat*(1-p0)
    #    p_hat[0] += p0
    
    p_hat,x,gamma = p_merge(p_hat,x*gamma,n_low, Y_pdf_high,Y_supp_high,n_high,x_step=0.001)
    
    if verbose:
        print('alpha',alpha)
        plt.figure()
        plot_density_1d(p_hat,x)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('dd_1d')
        plt.legend()
        plt.show() 
    return p_hat,x

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

def counts2pdf_1d(Y):
    Y_pdf=np.bincount(Y)
    Y_pdf=Y_pdf/Y.shape[0]
    Y_supp=np.arange(Y_pdf.shape[0])
    return Y_pdf,Y_supp

def Pmodel_cal(x,Y_supp,N_r):
    n_supp = x.shape[0]
    n_obs  = Y_supp.shape[0]
    P_model=sp.stats.poisson.pmf(np.repeat(np.reshape(np.arange(n_obs),[n_obs,1]),n_supp,axis=1),\
                                 np.repeat(np.reshape(x*N_r,[1,n_supp]),n_obs,axis=0))
    return P_model
               
def f_opt(alpha,Y_pdf,P_model,Q,c_reg):    
    l = l_cal(alpha,Y_pdf,P_model,Q,c_reg)
    grad = grad_cal(alpha,Y_pdf,P_model,Q,c_reg)
    #print('alpha',alpha)
    #print('grad',grad)
    #print('l',l,'\n')
    return -l,-grad

def px_cal(Q,alpha):
    P_X  = np.exp(Q.dot(alpha))
    P_X /= np.sum(P_X)
    return P_X
    
def l_cal(alpha,Y_pdf,P_model,Q,c_reg):
    P_X = px_cal(Q,alpha)
    l   = np.sum(Y_pdf*np.log(P_model.dot(P_X).clip(min=1e-18))) - c_reg*np.linalg.norm(alpha)
    return l    

def grad_cal(alpha,Y_pdf,P_model,Q,c_reg):    
    P_X = px_cal(Q,alpha) # P_X        
    P_Y = P_model.dot(P_X) # P_Y    
    W = (((P_model.T/(P_Y.clip(min=1e-18))).T-1)*P_X).T # gradient
    grad = Q.T.dot(W.dot(Y_pdf)) 
    grad = grad - c_reg*alpha/np.linalg.norm(alpha)
    return grad
               
def p_merge(p1,x1,n1,p2,x2,n2,x_step=None):   
    ## only care about non-zero parts 
    x_c=x1[-1]
    if x_step is None:
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
            
    x = np.array(x)
    p = np.array(p)
    p = p/np.sum(p)
    gamma = np.max(x)
    x = x/gamma

    return p,x,gamma


"""
    calculate some basic distributional quantities 
"""
def basic_cal(M):
    mean_ = M[:,0]
    var_ = (M[:,1]-M[:,0]**2).clip(min=0)
    cv_ = np.sqrt(var_) / mean_
    fano_ = var_/mean_
    return mean_,var_,cv_,fano_

def M_to_var(M):
    var_ = (M[1]-M[0]**2).clip(min=1e-18)
    return var_

def M_to_cv(M):
    var_ = (M[1]-M[0]**2).clip(min=1e-18)
    cv_ = np.sqrt(var_)/M[0]
    return cv_

def cov_to_PC(cov_,shrink=0):
    G = cov_.shape[0]
    diag_cov_ = np.diag(cov_) + shrink
    np.fill_diagonal(cov_,diag_cov_.clip(min=1e-12))  
    
    std_ = np.sqrt(diag_cov_)    
    std_ = std_.reshape([G,1])
    PC_= cov_/(std_.dot(std_.T))
    PC_ = PC_.clip(min=-1,max=1)
    return PC_

def zero_to_one(zero_matrix,shrink=0):
    G = zero_matrix.shape[0]
    p0 = np.diag(zero_matrix)
    one_matrix = np.ones([G,G])
    one_matrix = one_matrix - p0
    one_matrix = (one_matrix.T - p0).T
    one_matrix += zero_matrix
    np.fill_diagonal(one_matrix,1-p0)
    return one_matrix

# first row: theta
# second row: k(r)
def M_to_gamma(M):
    gamma_param = np.zeros([2,M.shape[1]])
    gamma_param[0] = (M[1]/M[0]-M[0]).clip(min=0)
    gamma_param[1] = (M[0]**2/(M[1]-M[0]**2)).clip(min=0,max=10)   
    return gamma_param

def get_rank(x):
    n_x = x.shape[0]
    rank = np.zeros([n_x],dtype=int)
    rank[np.argsort(x)] = np.arange(n_x)[::-1]
    return rank

def preprocess(data):
    data_ = data.copy()
    A = data_.X.data
    cap = np.percentile(A,99)
    A = A.clip(max=cap)
    data_.X.data = A
    return data_

"""
    plot the PCA score
"""

def plot_PCA_score(data,v1,v2,label=None,n_cell=None,ref_score=None,return_score=False,color='navy'):
    if n_cell is not None:
        subsample_index = np.random.choice(np.arange(data.shape[0]),n_cell,replace=False)  
    X = np.array(data.X.todense())
    X -= X.mean(axis=0)
    score1 = X.dot(v1)
    score2 = X.dot(v2)
    
    if ref_score is not None:
        ref_score1 = ref_score[0]
        ref_score2 = ref_score[1]
        if np.linalg.norm(ref_score1-score1)>np.linalg.norm(ref_score1+score1):
            score1 = -score1
        if np.linalg.norm(ref_score2-score2)>np.linalg.norm(ref_score2+score2):
            score2 = -score2
    
    plt.scatter(score1,score2,s=16,alpha=0.3,color=color)
    if return_score:
        return [score1,score2]
    
"""
    xx plot: comparing two estimates of the same quantity
"""

def plot_xx(x1,x2,logscale=False,alpha=0.4,s=4,color='orange',\
            xlabel=None,ylabel=None,lim=None,require_count=True):   
    
    if logscale:
        x1 = np.log10(x1.clip(min=1e-6))
        x2 = np.log10(x2.clip(min=1e-6))
        idx_select = (x1>-6)*(x2>-6)
        x1 = x1[idx_select]
        x2 = x2[idx_select]
        
        
    n_above = np.sum(x1<=x2)
    n_below = np.sum(x1>x2)
    
    if lim is None:        
        min_ = min(np.percentile(x1[x1>-6],1),np.percentile(x2[x2>-6],1)) - 0.5
        max_ = max(np.percentile(x1[x1>-6],99),np.percentile(x2[x2>-6],99)) + 0.5        
        x_min,y_min = min_,min_
        x_max,y_max = max_,max_           
    else:
        x_min,x_max,y_min,y_max = lim
        
    x_range = x_max - x_min
    y_range = y_max - y_min
        
    plt.scatter(x1,x2,alpha=alpha,color=color,s=s)
    plt.plot([x_min,x_max],[x_min,x_max],color='r',lw=1,zorder=10)
    plt.xlim([x_min,x_max])    
    plt.ylim([y_min,y_max])
    
    if require_count:
        plt.annotate('above:%d'%(n_above),[x_min+0.05*x_range,y_max-0.1*y_range])
        plt.annotate('below:%d'%(n_below),[x_max-0.35*x_range,y_min+0.05*y_range])
    
    if xlabel is not None:
        plt.xlabel(xlabel,fontsize=16) 
            
    if ylabel is not None: 
        plt.ylabel(ylabel,fontsize=16)
            
        
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
    
def fig_xx_zero(p0_dd1,p0_ml1,p0_dd2,p0_ml2,gene_list1,\
                gene_list2,data_name,mean_fil=0.1,s=20,margin=0.5):
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
    
"""
    Given the error dic with the parameters as the key, plot the tradeoff curve
    The keys are encoded as (B_sub,Nr_bar,Nr,Nc,i_rep)
"""
def plot_tradeoff_curve(error_ml_dic,error_dd_dic,output_folder=None,suffix='',ylabel='',title='',ann_ml=None,ann_dd=None,\
                       annotation=None,x_max=10,ann_offset=-3,ann_step=0.3,figsize=[6,5],B_trun_list=[]):
    B_list = []
    Nr_bar_list = []
    CI = 3
    for key_ in error_ml_dic.keys():
        B,Nr_bar,_,_ = key_to_param(key_)
        if B not in B_list: B_list.append(B)
        if Nr_bar not in Nr_bar_list: Nr_bar_list.append(Nr_bar)
    B_list = np.sort(np.array(B_list))
    Nr_bar_list = np.sort(np.array(Nr_bar_list))  
    n_Nr = Nr_bar_list.shape[0]
    
    plt.figure(figsize=figsize)
    for i_B,B in enumerate(B_list):
        if B in B_trun_list:
            continue
        
        err_ml = np.zeros([n_Nr])+100
        std_ml = np.zeros([n_Nr])+100
        err_dd = np.zeros([n_Nr])+100
        std_dd = np.zeros([n_Nr])+100
        
        for i_Nr,val_Nr in enumerate(Nr_bar_list):
            for key_ in error_ml_dic.keys():
                B_,Nr_bar_,_,Nc_ = key_to_param(key_)                
                if B == B_ and Nr_bar_ == val_Nr:
                    temp = len(error_ml_dic[key_])
                    B_bar_ = Nc_*Nr_bar
                    
                    idx_select = np.isfinite(np.array(error_ml_dic[key_]))
                    
                    
                    err_ml[i_Nr] = np.array(error_ml_dic[key_])[idx_select].mean()
                    std_ml[i_Nr] = np.array(error_ml_dic[key_])[idx_select].std() / np.sqrt(temp)
                    err_dd[i_Nr] = np.array(error_dd_dic[key_])[idx_select].mean()
                    std_dd[i_Nr] = np.array(error_dd_dic[key_])[idx_select].std() / np.sqrt(temp)
                    
        #plt.plot(Nr_bar_list,err_ml,label='ml B=%dk'%int(B/1000),marker='o',color='steelblue')
        Nr_bar_list = np.array(Nr_bar_list)
        idx_select = (err_ml<100)
        
        plt.plot(np.log10(Nr_bar_list[idx_select]),err_ml[idx_select],marker='o',color='steelblue')
        plt.fill_between(np.log10(Nr_bar_list[idx_select]),err_ml[idx_select]-CI*std_ml[idx_select],\
                         err_ml[idx_select]+CI*std_ml[idx_select],alpha=0.4,color='steelblue')
        if ann_ml is not None:
            plt.annotate('B_bar=%0.1fk'%(B_bar_/1000),[np.log10(Nr_bar_list[ann_ml]),err_ml[ann_ml]])
        #plt.plot(Nr_bar_list,err_dd,label='dd B=%dk'%int(B/1000),marker='o',color='orange')        
        plt.plot(np.log10(Nr_bar_list[idx_select]),err_dd[idx_select],marker='o',color='orange')        
        plt.fill_between(np.log10(Nr_bar_list[idx_select]),err_dd[idx_select]-CI*std_dd[idx_select],\
                         err_dd[idx_select]+CI*std_dd[idx_select],alpha=0.4,color='orange')
        
        if ann_dd is not None:
            plt.annotate('$B_%d$'%(i_B+1),[np.log10(Nr_bar_list[ann_dd]),err_dd[ann_dd]])
            
        if (ann_ml is None) and (ann_dd is None):
            plt.annotate('$B_%d$'%(i_B+1),[np.log10(0.02),ann_offset-ann_step*i_B])
            
        #if ann_dd is not None:
        #    plt.annotate('B_bar=%0.1fk'%(B_bar_/1000),[np.log10(Nr_bar_list[ann_dd]),err_dd[ann_dd]])
        #    
        #if (ann_ml is None) and (ann_dd is None):
        #    plt.annotate('B_bar=%0.1fk'%(B_bar_/1000),[np.log10(0.02),ann_offset-ann_step*i_B])
    
    #plt.xticks(np.arange(x_max+0.1, step=1))
    #plt.xlim([-0.1,x_max+0.1])
    if annotation is not None:
        for annon in annotation:
            text, loc = annon
            plt.annotate(text,loc)
            plt.scatter([loc[0]],[loc[1]],marker='^',color='red')
    
    plt.plot(np.log10(Nr_bar_list[-1]),err_ml[-1],label='Plug-in',marker='o',color='steelblue')
    plt.plot(np.log10(Nr_bar_list[-1]),err_dd[-1],label='EB',marker='o',color='orange')

    plt.xticks(np.log10(Nr_bar_list),Nr_bar_list)  
    plt.grid(linestyle='dashed')
    plt.xlabel('mean reads (per cell per gene)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_folder+'/tradeoff_curve_'+suffix+'.png')
    plt.savefig(output_folder+'/tradeoff_curve_'+suffix+'.pdf')
    plt.show()
    plt.close('all')
    
def plot_tradeoff_posthoc_curve(error_ml_dic,error_dd_dic,output_folder=None,\
                                suffix='',ylabel='',title='',ann_ml=None,ann_dd=None,\
                                x_max=10,ann_offset=-3,ann_step=0.3,figsize=[6,5]):
    def key2param(str_):
        temp = str_.split('_')
        Nr_bar = float(temp[0])
        Nr = float(temp[1])
        Nc = int(temp[2])
        return Nr_bar,Nr,Nc
    def param2key(Nr_bar,Nr,Nc):
        return '%0.2f_%0.2f_%d'%(Nr_bar,Nr,Nc)
    
    Nc_list = []
    Nr_bar_list = []
    CI = 3
    for key_ in error_ml_dic.keys():
        Nr_bar,Nr,Nc = key2param(key_)
        if Nc not in Nc_list: Nc_list.append(Nc)
        if Nr_bar not in Nr_bar_list: Nr_bar_list.append(Nr_bar)
    Nc_list = np.sort(np.array(Nc_list))
    Nr_bar_list = np.sort(np.array(Nr_bar_list))  
    n_Nr = Nr_bar_list.shape[0]
    
    plt.figure(figsize=figsize)
    for i_Nc,Nc in enumerate(Nc_list):
        err_ml = np.zeros([n_Nr])
        std_ml = np.zeros([n_Nr])
        err_dd = np.zeros([n_Nr])
        std_dd = np.zeros([n_Nr])
        
        for i_Nr,val_Nr in enumerate(Nr_bar_list):
            for key_ in error_ml_dic.keys():
                Nr_bar_,Nr_,Nc_ = key2param(key_)
                if Nc == Nc_ and Nr_bar_ == val_Nr:
                    temp = len(error_ml_dic[key_])
                    err_ml[i_Nr] = np.array(error_ml_dic[key_]).mean()
                    std_ml[i_Nr] = np.array(error_ml_dic[key_]).std() / np.sqrt(temp)
                    err_dd[i_Nr] = np.array(error_dd_dic[key_]).mean()
                    std_dd[i_Nr] = np.array(error_dd_dic[key_]).std() / np.sqrt(temp)
                    
        plt.plot(np.log10(Nr_bar_list),err_ml,marker='o',color='steelblue')
        plt.fill_between(np.log10(Nr_bar_list),err_ml-CI*std_ml,err_ml+CI*std_ml,alpha=0.4,color='steelblue')
        if ann_ml is not None:
            plt.annotate('$N_c$=%dk'%int(Nc/1000),[np.log10(Nr_bar_list[ann_ml]),err_ml[ann_ml]])
        plt.plot(np.log10(Nr_bar_list),err_dd,marker='o',color='orange')        
        plt.fill_between(np.log10(Nr_bar_list),err_dd-CI*std_dd,err_dd+CI*std_dd,alpha=0.4,color='orange')
        if ann_dd is not None:
            plt.annotate('$N_c$=%dk'%int(Nc/1000),[np.log10(Nr_bar_list[ann_dd]),err_dd[ann_dd]])
            
        if (ann_ml is None) and (ann_dd is None):
            plt.annotate('$N_c$=%dk'%int(Nc/1000),[np.log10(0.02),ann_offset-ann_step*i_Nc])
    
    plt.plot(np.log10(Nr_bar_list),err_ml,label='Plug-in',marker='o',color='steelblue')
    plt.plot(np.log10(Nr_bar_list),err_dd,label='EB',marker='o',color='orange')

    plt.xticks(np.log10(Nr_bar_list),Nr_bar_list)  
    plt.grid(linestyle='dashed')
    plt.xlabel('Mean read counts (per cell per gene)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_folder+'/tradeoff_curve_'+suffix+'.png')
    plt.savefig(output_folder+'/tradeoff_curve_'+suffix+'.pdf')
    plt.show()
    plt.close('all')
    

def fname_to_key(fname):
    fname = os.path.splitext(fname)[0]
    fname = fname.strip().split('_')[1:]
    B,Nr_bar,Nr,Nc,i_rep = fname
    key_ = B+'_'+Nr_bar+'_'+Nr+'_'+Nc
    return key_,int(i_rep)

## The keys are encoded as (B_sub,Nr_bar,Nr,Nc,i_rep)
def key_to_param(key):
    key = key.strip().split('_')
    return int(key[0]),float(key[1]),float(key[2]),int(key[3])

def error_p0(p0,p0_true):
    p0 = p0.clip(min=1e-2)
    p0_true = p0_true.clip(min=1e-2)
    err = np.absolute(p0-p0_true)
    err = np.mean(err)
    return err

def cosine_distance(v1,v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    #d = 1-(v1.dot(v2))**2
    d = 1-np.absolute(v1.dot(v2))
    return d

def d_PC(PC1,PC2):
    d = 0 
    ct = 0
    n = PC1.shape[0]
    for i in range(n):
        for j in range(i+1,n):
            d += (PC1[i,j]-PC2[i,j])**2
            ct += 1
    return d/ct


"""
    Calculate the cumulative Nr_bar information of a dataset 
"""
def get_fingerprint(mean_count):
    mean_count = np.sort(mean_count)[::-1]
    finger_print = []
    quantile_list = [0,1000,2000,3000,4000]
    for i in range(len(quantile_list)-1):
        finger_print.append(np.mean(mean_count[quantile_list[i]:quantile_list[i+1]]))
    finger_print = np.array(finger_print)
    return finger_print

def get_fingerprint_2(mean_count):
    mean_count = np.sort(mean_count)[::-1]
    finger_print = []
    quantile_list = [100,500,1000,1500,2000,2500,3000]
    for i in quantile_list:
        finger_print.append(mean_count[i])
    finger_print = np.array(finger_print)
    return finger_print