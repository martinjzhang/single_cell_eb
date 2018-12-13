""" 
    sceb_extra: "Single Cell Density Deconvolution"
    Extra functions accompanying the paper 'One read per cell is optimal for Single-Cell RNA-Seq'
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
import sceb.scdd as sd

from sceb.b_spline_nd import *
from sceb.util import *

def subsample_single_gene(Y, Nc_new, B, random_state=0):
    success_flag = True
    Y = np.array(Y, dtype=int)
    np.random.seed(random_state)
    # Subsample cells 
    ind_cell = np.random.permutation(Y.shape[0])[:Nc_new]
    temp = Y[ind_cell]
    # Subsample reads 
    read_list = []
    for i in range(Nc_new):
        read_list.extend([i]*temp[i])
    read_list = np.array(read_list, dtype=int)
    rand_ind = np.random.permutation(read_list.shape[0])[:B]
    if read_list.shape[0]<B:
        print('not enough reads! ')
        success_flag = False
    read_list_new = read_list[rand_ind]
    Y_new = np.zeros([Nc_new])
    for i in read_list_new:
        Y_new[i] = Y_new[i] + 1
    return Y_new, ind_cell, success_flag

def eb_1d_moment(data, size_factor=None, verbose=False, k=2, Nr=1):
    """Calculate the moments using plug-in (ml) and EB (dd)
    
    Args: 
        data (AnnData): The scRNA-Seq CG (cell-gene) matrix.
        size_factor ((Nc,) ndarray): the cell size factor.
        k (int): the number of moments to estimate.
        
    Returns:
        M_ml ((k,G) ndarray): the plug-in estimates of the moments.
        M_dd ((k,G) ndarray): the EB estimates of the moments.
    """    
    if k>4:
        print('## The program only outputs at most 4 moments')        
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc,G))
    # Initialization
    X = data.X ## csr file   
    M_ml = np.zeros([k,G], dtype=float)
    M_dd = np.zeros([k,G], dtype=float)
    # Moments of size factor
    size_moment = np.ones([k], dtype=float)
    if size_factor is not None:
        row_weight = 1/size_factor
        for i in range(k):
            size_moment[i] = np.mean(size_factor**(i+1))
    else:
        row_weight = np.ones([Nc], dtype=int)
    # EB moment estimation
    M1_ = scipy.sparse.csr_matrix.mean(X.power(1), axis=0)
    M2_ = scipy.sparse.csr_matrix.mean(X.power(2), axis=0)
    M3_ = scipy.sparse.csr_matrix.mean(X.power(3), axis=0)
    M4_ = scipy.sparse.csr_matrix.mean(X.power(4), axis=0)
    if k>=1:
        temp = sp.sparse.csc_matrix.dot(row_weight.reshape([1,-1]),X)
        M_ml[0] = np.array(temp).reshape(-1)/Nc
        M_dd[0] = M1_/size_moment[0]
    if k>=2:
        temp = sp.sparse.csc_matrix.dot(row_weight.reshape([1,-1])**2,X.power(2))
        M_ml[1] = np.array(temp).reshape(-1)/Nc
        M_dd[1] = (M2_ - M1_)/size_moment[1]
    if k>=3:
        temp = sp.sparse.csc_matrix.dot(row_weight.reshape([1,-1])**3,X.power(3))
        M_ml[2] = np.array(temp).reshape(-1)/Nc
        M_dd[2] = (M3_ - 3*M2_ + 2*M1_ )/size_moment[2]
    if k>=4:
        temp = sp.sparse.csc_matrix.dot(row_weight.reshape([1,-1])**4,X.power(3))
        M_ml[3] = np.array(temp).reshape(-1)/Nc
        M_dd[3] = (M4_ - 6*M3_ + 11*M2_ - 6*M1_)/size_moment[3]
    # Normalize by Nr
    for i in range(k):
        M_ml[i] = M_ml[i]/Nr**(i+1)
        M_dd[i] = M_dd[i]/Nr**(i+1)            
    if verbose: 
        print('#total: %0.2fs'%(time.time()-start_time))
    return M_ml,M_dd

def eb_PC(data, size_factor=None, verbose=False, Nr=1):
    """ EB estimation of the covariance matrix and the Pearson correlation matrix.
    
    Args:
        data (AnnData): the scRNA-Seq CG (cell-gene) matrix, 
            whose data is stored as a CSR sparse matrix.
        size_factor ((Nc,) ndarray): the cell size_factor.
        PC_prune (bool): if set the value to be zero for genes with small EB estimate variance (
            due to stability consideration)
        
    Returns:
        mean_dd ((G,) ndarray): the mean gene expression level.
        cov_dd ((G,G) ndarray): the estimated covariance matrix.
        PC_dd ((G,G) ndarray): the estimated Pearson correlation matrix.
    """    
    if verbose: 
        start_time=time.time()
        print('#time start: 0.0s')
    Nc,G = data.shape
    # Nr = data.X.sum()/Nc    
    if verbose: 
        print('n_cell=%d, n_gene=%d'%(Nc, G))   
    X = data.X ## csr file
    # Moments of size factor
    size_moment = np.ones([2], dtype=float)
    if size_factor is not None:
        row_weight = 1/size_factor
        for i in range(2):
            size_moment[i] = np.mean(size_factor**(i+1))
    else:
        row_weight = np.ones([Nc], dtype=int)
    # Compute the M1 and M2 moment
    M_ml,M_dd = eb_1d_moment(data, size_factor=size_factor, k=2, Nr=Nr)
    # Compute the M11 moment
    X_rw = assign_row_weight(X,row_weight)
    M11_ml = np.array((X_rw.transpose().dot(X_rw)/Nc).todense())
    M11_dd = np.array((X.transpose().dot(X)/Nc).todense())/size_moment[1]
    np.fill_diagonal(M11_dd, M_dd[1, :])
    # Convert to PC: ml
    temp_mean = M_ml[0,:].reshape([-1,1])
    temp_std = np.sqrt((M_ml[1,:].reshape([-1,1]) - temp_mean**2).clip(min=0))
    PC_ml = (M11_ml - temp_mean.dot(temp_mean.T)) / (temp_std.dot(temp_std.T))
    # Convert to PC: dd
    temp_mean = M_dd[0,:].reshape([-1,1])
    temp_std = np.sqrt((M_dd[1,:].reshape([-1,1]) - temp_mean**2).clip(min=0))
    PC_dd = (M11_dd - temp_mean.dot(temp_mean.T)) / (temp_std.dot(temp_std.T))    
    if verbose: 
        print('#total: %0.2fs'%(time.time()-start_time))
    PC_ml = PC_ml.clip(min=-1, max=1)
    PC_dd = PC_dd.clip(min=-1, max=1)
    return PC_ml,PC_dd

def assign_row_weight(X,row_weight): 
    X = X.astype(np.float64)
    X = (X.T.multiply(row_weight)).T.tocsr()
    return X

def M_diagnosis(Y1, Y2, size_factor=None):
    if size_factor is None:
        size_factor = np.ones([Y1.shape[0]], dtype=float)
    # Plug-in
    Y1_ml = Y1/size_factor
    Y2_ml = Y2/size_factor
    PC_ml = np.corrcoef(Y1_ml, Y2_ml)[0,1]
    print('# Plug-in: mean_1=%0.3f, mean_2=%0.3f, M2_1=%0.3f, M2_2=%0.3f, var_1=%0.3f, var_2=%0.3f, cov=%0.3f, PC=%0.3f'%
          (np.mean(Y1_ml), np.mean(Y2_ml), np.mean(Y1_ml**2), np.mean(Y2_ml**2), np.var(Y1_ml), np.var(Y2_ml),
           np.cov(Y1_ml, Y2_ml)[0,1], PC_ml))   
    # EB 
    mean_1 = np.mean(Y1) / np.mean(size_factor)
    mean_2 = np.mean(Y2) / np.mean(size_factor)
    M2_1 = (np.mean(Y1**2) - np.mean(Y1)) / np.mean(size_factor**2)
    M2_2 = (np.mean(Y2**2) - np.mean(Y2)) / np.mean(size_factor**2)
    # M2_1 = (np.mean(Y1**2)) / np.mean(size_factor**2)
    # M2_2 = (np.mean(Y2**2)) / np.mean(size_factor**2)
    var_1 = (M2_1 - mean_1**2).clip(min=0)
    var_2 = (M2_2 - mean_2**2).clip(min=0)
    M11 = np.mean(Y1*Y2) / np.mean(size_factor**2)
    cov11 = M11 - mean_1*mean_2
    PC_dd = cov11 / np.sqrt(var_1*var_2)
    PC_dd = min(1, PC_dd)
    PC_dd = max(-1, PC_dd)
    print('# EB: mean_1=%0.3f, mean_2=%0.3f, M2_1=%0.3f, M2_2=%0.3f, var_1=%0.3f, var_2=%0.3f, cov=%0.3f, PC=%0.3f'%
          (mean_1, mean_2, M2_1, M2_2, M2_1-mean_1**2, M2_2-mean_2**2,cov11, PC_dd))
    # return PC_ml, PC_dd
def M_single(Y, size_factor=None, size_moment=None, Nr=1):
    if size_factor is None:
        size_factor = np.ones([Y.shape[0]], dtype=float)
    if size_moment is None:
        size_moment = np.zeros([2], dtype=float)
        size_moment[0] = np.mean(size_factor)
        size_moment[1] = np.mean(size_factor**2)
    # Plug-in estimate
    M_ml = np.zeros([2], dtype=float)
    Y_ml = Y/size_factor
    M_ml[0] = np.mean(Y_ml)/Nr
    M_ml[1] = np.mean(Y_ml**2)/Nr**2
    # Eb estimate
    M_dd = np.zeros([2], dtype=float)
    M_dd[0] = np.mean(Y) / size_moment[0] / Nr
    M_dd[1] = (np.mean(Y**2) - np.mean(Y)) / size_moment[1] / Nr**2
    return M_ml, M_dd

def cv_single(Y, size_factor=None, size_moment=None):
    if size_factor is None:
        size_factor = np.ones([Y.shape[0]], dtype=float)
    if size_moment is None:
        size_moment = np.zeros([2], dtype=float)
        size_moment[0] = np.mean(size_factor)
        size_moment[1] = np.mean(size_factor**2)
    # Plug-in
    Y_ml = Y/size_factor
    cv_ml = np.std(Y_ml) / np.mean(Y_ml)
    # EB 
    mean_ = np.mean(Y) / size_moment[0]
    M2_ = (np.mean(Y**2) - np.mean(Y)) / size_moment[1]
    var_ = (M2_ - mean_**2).clip(min=0)
    cv_dd = np.sqrt(var_)/mean_
    return cv_ml, cv_dd

def cv_single_fish(Y, size_factor=None, size_moment=None):
    if size_factor is None:
        size_factor = np.ones([Y.shape[0]], dtype=float)
    if size_moment is None:
        size_moment = np.zeros([2], dtype=float)
        size_moment[0] = np.mean(size_factor)
        size_moment[1] = np.mean(size_factor**2)
    mean_ = np.mean(Y) / size_moment[0]
    M2_ = np.mean(Y**2) / size_moment[1]
    var_ = (M2_ - mean_**2).clip(min=0)
    cv_fish = np.sqrt(var_)/mean_
    return cv_fish

def p0_single(Y, size_factor=None, relative_depth=1):
    Y_ann = to_AnnData(np.tile(Y.reshape([-1,1]),2))
    p0_ml,p0_dd = sd.dd_inactive_prob(Y_ann, size_factor=size_factor,
                                      relative_depth=relative_depth, verbose=False)
    return p0_ml[0],p0_dd[0]

def PC_single(Y1, Y2, size_factor=None, size_moment=None):
    if size_factor is None:
        size_factor = np.ones([Y1.shape[0]], dtype=float)
    if size_moment is None:
        size_moment = np.zeros([2], dtype=float)
        size_moment[0] = np.mean(size_factor)
        size_moment[1] = np.mean(size_factor**2)
    # Plug-in
    Y1_ml = Y1/size_factor
    Y2_ml = Y2/size_factor
    PC_ml = np.corrcoef(Y1_ml, Y2_ml)[0,1]
    # EB 
    mean_1 = np.mean(Y1) / size_moment[0]
    mean_2 = np.mean(Y2) / size_moment[0]
    M2_1 = (np.mean(Y1**2) - np.mean(Y1)) / size_moment[1]
    M2_2 = (np.mean(Y2**2) - np.mean(Y2)) / size_moment[1]
    var_1 = (M2_1 - mean_1**2).clip(min=0)
    var_2 = (M2_2 - mean_2**2).clip(min=0)
    M11 = np.mean(Y1*Y2) / size_moment[1]
    cov11 = M11 - mean_1*mean_2
    PC_dd = cov11 / np.sqrt(var_1*var_2)
    PC_dd = min(1, PC_dd)
    PC_dd = max(-1, PC_dd)
    return PC_ml, PC_dd

def PC_single_fish(Y1, Y2, size_factor=None, size_moment=None):
    if size_factor is None:
        size_factor = np.ones([Y1.shape[0]], dtype=float)
    if size_moment is None:
        size_moment = np.zeros([2], dtype=float)
        size_moment[0] = np.mean(size_factor)
        size_moment[1] = np.mean(size_factor**2)
    # EB 
    mean_1 = np.mean(Y1) / size_moment[0]
    mean_2 = np.mean(Y2) / size_moment[0]
    M2_1 = np.mean(Y1**2) / size_moment[1]
    M2_2 = np.mean(Y2**2) / size_moment[1]
    var_1 = (M2_1 - mean_1**2).clip(min=0)
    var_2 = (M2_2 - mean_2**2).clip(min=0)
    M11 = np.mean(Y1*Y2) / size_moment[1]
    cov11 = M11 - mean_1*mean_2
    PC_fish = cov11 / np.sqrt(var_1*var_2)
    PC_fish = min(1, PC_fish)
    PC_fish = max(-1, PC_fish)
    return PC_fish

def PC_heatmap(PC, gene_list): 
    def strip0(s):
        if s[0]=='0':
            return s[1:]
        elif s[0] == '-' and s[1] =='0':
            return '-'+s[2:]
        else:
            return s
    plt.imshow(PC,vmin=-1,vmax=1)  
    n_gene = gene_list.shape[0]
    plt.xticks(np.arange(n_gene),gene_list,rotation=90,fontsize=15)
    plt.yticks(np.arange(n_gene),gene_list,fontsize=15)
    for i in range(n_gene):
        for j in range(n_gene):
            text = plt.text(j,i,'%s'%strip0('%0.2f'%PC[i,j]),
                           ha="center", va="center", color="w",fontsize=12)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)

def to_AnnData(Y, gene_list=None):
    """ Convert a ndarray to AnnData with sparse csr reads
    """
    Y = sp.sparse.csr_matrix(Y)
    if gene_list is None:
        gene_list = []
        for i in range(Y.shape[1]):
            gene_list.append('gene %d'%(i))        
    var = pd.DataFrame(index=gene_list)
    data = sc.AnnData(Y,var=var)
    return data

def simulate_tradeoff_cv_single(Y_gene, gene, B_list, size_factor, n_param=10, n_rep=100):
    """ Simulate the tradeoff curve for a single gene
    """
    n_cell = Y_gene.shape[0]
    res = {}
    print(gene, 'Nr_bar=%0.3f'%(np.mean(Y_gene)))
    for B in B_list:
        res[B] = {}
        Nr_bar_new_lb = 1.01*B/n_cell
        Nr_bar_new_ub = min(0.7*np.mean(Y_gene), 1)
        Nr_bar_new_list = np.exp(np.linspace(np.log(Nr_bar_new_lb),
                                             np.log(Nr_bar_new_ub), n_param))
        res[B]['cv_ml'] = np.zeros([n_rep, n_param])
        res[B]['cv_dd'] = np.zeros([n_rep, n_param])
        res[B]['Nr_bar_new_list'] = Nr_bar_new_list
        for i_param,Nr_bar_new in enumerate(Nr_bar_new_list):
            print('B=%d, Nr_bar_new=%0.3f'%(B,Nr_bar_new))
            Nc_new = int(B/Nr_bar_new)
            for i_rep in range(n_rep): 
                Y_new,ind_cell,flag = subsample_single_gene(Y_gene, Nc_new,\
                                                                B, random_state=i_rep)
                if flag:
                    size_factor_new = size_factor[ind_cell]
                    cv_ml,cv_dd = cv_single(Y_new, size_factor=size_factor_new)
                    res[B]['cv_ml'][i_rep, i_param] = cv_ml
                    res[B]['cv_dd'][i_rep, i_param] = cv_dd   
                else:
                    res[B]['cv_ml'][i_rep, i_param] = np.nan
                    res[B]['cv_dd'][i_rep, i_param] = np.nan   
    return res

def simulate_tradeoff_p0_single(Y_gene, gene, B_list, size_factor, n_param=10, n_rep=100,
                                target_Nr = 1):
    """ Simulate the tradeoff curve for a single gene
    """
    n_cell = Y_gene.shape[0]
    res = {}
    print(gene, 'Nr_bar=%0.3f'%(np.mean(Y_gene)))
    for B in B_list:
        res[B] = {}
        Nr_bar_new_lb = 1.01*B/n_cell
        Nr_bar_new_ub = min(0.7*np.mean(Y_gene), 1)
        Nr_bar_new_list = np.exp(np.linspace(np.log(Nr_bar_new_lb),
                                             np.log(Nr_bar_new_ub), n_param))
        res[B]['p0_ml'] = np.zeros([n_rep, n_param])
        res[B]['p0_dd'] = np.zeros([n_rep, n_param])
        res[B]['Nr_bar_new_list'] = Nr_bar_new_list
        for i_param,Nr_bar_new in enumerate(Nr_bar_new_list):
            print('B=%d, Nr_bar_new=%0.3f'%(B,Nr_bar_new))
            Nc_new = int(B/Nr_bar_new)
            for i_rep in range(n_rep): 
                Y_new,ind_cell,flag = subsample_single_gene(Y_gene, Nc_new,\
                                                            B, random_state=i_rep)
                if flag:
                    size_factor_new = size_factor[ind_cell]
                    p0_ml,p0_dd = p0_single(Y_new, size_factor = size_factor_new,
                                            relative_depth = Y_new.mean()/target_Nr)
                    res[B]['p0_ml'][i_rep, i_param] = p0_ml
                    res[B]['p0_dd'][i_rep, i_param] = p0_dd   
                else:
                    res[B]['p0_ml'][i_rep, i_param] = np.nan
                    res[B]['p0_dd'][i_rep, i_param] = np.nan   
    return res