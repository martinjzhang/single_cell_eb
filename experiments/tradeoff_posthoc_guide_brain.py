import logging
import os
from sys import argv

#ID = argv[1]
#G = int(argv[2])
#n_rep = int(argv[3])

output_folder = '/home/martin/single_cell_eb/figures/figure_tradeoff_posthoc_brain'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logging.basicConfig(level=logging.INFO,\
        format='%(module)s:: %(message)s',\
        filename=output_folder+'/result.log', filemode='w')
logger = logging.getLogger(__name__)

n_rep = 20
Nc_list = [1000,5000,10000,30000,70000]
Nr_bar_list = [0.02,0.05,0.1, 0.2, 0.5, 1, 2, 5, 10]

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import h5py
import time
import pickle


import scanpy.api as sc
import sceb.data_loader as dl
import sceb.scdd as sd
from sceb.util import *

def main():
    start_time = time.time()
    ## Read subsample parameters
    fname = '/home/martin/single_cell_eb/figures/figure_tradeoff_curve/subsample_param.pickle'
    f_myfile = open(fname, 'rb')
    B_sub_list = pickle.load(f_myfile)
    subsample_param_dic = pickle.load(f_myfile)
    f_myfile.close()
    
    ## toy example from 9k 
    data = dl.load_10x_9k()
    mean_count = np.array(data.X.mean(axis=0)).reshape(-1)
    sort_idx = np.argsort(mean_count)[::-1]
    
    marker_gene_list= ['Lhx6','Rpp25','Npy','Sst','Nxph1','Gria1','Gria2','Dlx2','Dlx1','Cldn1','Col4a2','Col4a1',\
                  'Itm2a','Ccl4','Neurod1','Vtn','Igfbp4','Gadd45g','Hs6st2','Satb2','Jakmip1','Clmp','Nrp1',\
                  'S100a10','Pcp4','Rnd2','Mef2c','Ptprd','Ly6h','Camk2b','Meg3','Tbr1','Reln',\
                  'Cdk1','Cdkn2c','Pax6','Prox1']

    gene_list = []
    for gene in marker_gene_list:
        if gene in data.var_names:
            if data[:,gene].X.mean()>0.6:
                print(gene,data[:,gene].X.mean())
                gene_list.append(gene)
                Y = np.array(data[:,gene].X)
    G = len(gene_list)
                
                
    ## Generate the toy example
    Y = np.array(data[:,gene_list].X.todense()).astype(dtype=int)

    p_true = np.ones([Y.shape[0]])
    p_true = p_true / np.sum(p_true)

    ## Making each column of x to have the same mean 
    x_true = Y / Y.mean(axis=0)
    x_true = x_true.clip(max=np.percentile(x_true,99))
    x_true = x_true/x_true.mean(axis=0)
    x_true = x_true/x_true.sum()*Y.shape[0]

    ## calculate the related quantities
    M_true = np.zeros([2,x_true.shape[1]])
    M_true[0] = np.mean(x_true,axis=0)
    M_true[1] = np.mean(x_true**2,axis=0)
    mean_true = x_true.mean(axis=0)
    var_true = x_true.var(axis=0)
    cv_true = np.sqrt(var_true)/mean_true
    gamma_param_true = sd.M_to_gamma(M_true)

    cov_true = np.cov(x_true.T)
    _,V_true = np.linalg.eigh(cov_true)
    PeCo_true = np.corrcoef(x_true.T)
    p0_true = np.mean(x_true==0,axis=0)
    kappa = G
    temp = np.exp(-kappa*x_true)
    inactive_true = np.mean(temp,axis=0)
    pairwise_inactive_true = temp.T.dot(temp)/temp.shape[0]
    np.fill_diagonal(pairwise_inactive_true,inactive_true)
    ##
    dist_true = {}
    for i in range(Y.shape[1]):
        p_dist_true_1d = np.bincount(Y[:,i])
        p_dist_true_1d = p_dist_true_1d / np.sum(p_dist_true_1d)
        x_dist_true_1d = np.arange(p_dist_true_1d.shape[0])
        dist_true[i] = [p_dist_true_1d,x_dist_true_1d]            
             
                
    ## start simulation
    temp_str = 'genelist: '
    for gene in gene_list:
        temp_str += gene+', '
    logger.info(temp_str)

    test_list = ['mean','var','cv','gamma_theta','gamma_r','cov','PC','PCA','zero','pw_zero','dist']

    ## use a two-layer dic to store the result
    err_ml_dic = {}
    err_dd_dic = {}

    for test_type in test_list:
        err_ml_dic[test_type] = {}
        err_dd_dic[test_type] = {}

    start_time = time.time()

    for Nc in Nc_list:
        for Nr_bar in Nr_bar_list:
            Nr = Nr_bar*G
            key_ = '%0.2f_%0.2f_%d'%(Nr_bar,Nr,Nc)
            for test_type in test_list:            
                err_ml_dic[test_type][key_] = []
                err_dd_dic[test_type][key_] = []

            for i_rep in range(n_rep):
                print('time %0.1f: '%(time.time()-start_time) + key_+'_%d'%i_rep)
                logger.info('time %0.1f: '%(time.time()-start_time) + key_+'_%d'%i_rep)
                
                ## Data generation
                data,size_factor = dl.poi_data_gen_nd(p_true,x_true,Nc=Nc,Nr=Nr,random_seed=i_rep)

                ## mean,var,cv 
                M_ml,M_dd = sd.dd_1d_moment(data,verbose=False,size_factor=size_factor)

                err_ml = np.log10(np.mean((mean_true-M_ml[0]/Nr)**2)) - np.log10(np.mean(mean_true**2))
                err_dd = np.log10(np.mean((mean_true-M_dd[0]/Nr)**2)) - np.log10(np.mean(mean_true**2))
                err_ml_dic['mean'][key_].append(err_ml)
                err_dd_dic['mean'][key_].append(err_dd)


                var_ml = sd.M_to_var(M_ml)
                var_dd = sd.M_to_var(M_dd)
                err_ml = np.log10(np.mean((var_true-var_ml/Nr**2)**2)) - np.log10(np.mean(var_true**2))
                err_dd = np.log10(np.mean((var_true-var_dd/Nr**2)**2)) - np.log10(np.mean(var_true**2))
                err_ml_dic['var'][key_].append(err_ml)
                err_dd_dic['var'][key_].append(err_dd)

                cv_ml = sd.M_to_cv(M_ml)
                cv_dd = sd.M_to_cv(M_dd)
                err_ml = np.log10(np.mean((cv_true-cv_ml)**2)) - np.log10(np.mean(cv_true**2))
                err_dd = np.log10(np.mean((cv_true-cv_dd)**2)) - np.log10(np.mean(cv_true**2))
                err_ml_dic['cv'][key_].append(err_ml)
                err_dd_dic['cv'][key_].append(err_dd)

                gamma_param_ml = sd.M_to_gamma(M_ml)
                gamma_param_ml[0] /= Nr
                gamma_param_dd = sd.M_to_gamma(M_dd)
                gamma_param_dd[0] /= Nr
                err_ml = np.log10(np.mean((gamma_param_true[0]-gamma_param_ml[0])**2))\
                         - np.log10(np.mean(gamma_param_true[0]**2))
                err_dd = np.log10(np.mean((gamma_param_true[0]-gamma_param_dd[0])**2))\
                         - np.log10(np.mean(gamma_param_true[0]**2))
                err_ml_dic['gamma_theta'][key_].append(err_ml)
                err_dd_dic['gamma_theta'][key_].append(err_dd)
                err_ml = np.log10(np.mean((gamma_param_true[1]-gamma_param_ml[1])**2))\
                         - np.log10(np.mean(gamma_param_true[1]**2))
                err_dd = np.log10(np.mean((gamma_param_true[1]-gamma_param_dd[1])**2))\
                         - np.log10(np.mean(gamma_param_true[1]**2))
                err_ml_dic['gamma_r'][key_].append(err_ml)
                err_dd_dic['gamma_r'][key_].append(err_dd)

                ## PC and PCA
                mean_ml,cov_ml,PeCo_ml = sd.ml_covariance(data,size_factor=size_factor,verbose=False)
                mean_dd,cov_dd,PeCo_dd = sd.dd_covariance(data,size_factor=size_factor,verbose=False,PC_prune=False)

                err_ml = np.log10(np.mean((cov_true-cov_ml/Nr**2)**2)) - np.log10(np.mean(cov_true**2))
                err_dd = np.log10(np.mean((cov_true-cov_dd/Nr**2)**2)) - np.log10(np.mean(cov_true**2))
                err_ml_dic['cov'][key_].append(err_ml)
                err_dd_dic['cov'][key_].append(err_dd)

                err_ml = np.log10(np.mean((PeCo_true-PeCo_ml)**2)) - np.log10(np.mean(PeCo_true**2))
                err_dd = np.log10(np.mean((PeCo_true-PeCo_dd)**2)) - np.log10(np.mean(PeCo_true**2))
                err_ml_dic['PC'][key_].append(err_ml)
                err_dd_dic['PC'][key_].append(err_dd)

                ## PCA
                _,V_ml = np.linalg.eigh(cov_ml)
                _,V_dd = np.linalg.eigh(cov_dd)
                err_ml = np.log10(sd.cosine_distance(V_true[:,-1],V_ml[:,-1]))
                err_dd = np.log10(sd.cosine_distance(V_true[:,-1],V_dd[:,-1]))            
                err_ml_dic['PCA'][key_].append(err_ml)
                err_dd_dic['PCA'][key_].append(err_dd)

                ## zero                        
                p0_ml,p0_dd = sd.dd_inactive_prob(data,relative_depth=Nr/kappa,size_factor=size_factor,verbose=False)
                err_ml = np.log10(np.mean((p0_ml-inactive_true)**2)) - np.log10(np.mean(inactive_true**2))
                err_dd = np.log10(np.mean((p0_dd-inactive_true)**2)) - np.log10(np.mean(inactive_true**2))
                err_ml_dic['zero'][key_].append(err_ml)
                err_dd_dic['zero'][key_].append(err_dd)

                ## pw zero
                pw_p0_ml,pw_p0_dd = sd.dd_pairwise_inactive_prob(data,relative_depth=Nr/kappa,\
                                                                 size_factor=size_factor,verbose=False)
                err_ml = np.log10(sd.d_PC(pairwise_inactive_true,pw_p0_ml)) - np.log10(np.mean(pairwise_inactive_true**2))
                err_dd = np.log10(sd.d_PC(pairwise_inactive_true,pw_p0_dd)) - np.log10(np.mean(pairwise_inactive_true**2))
                err_ml_dic['pw_zero'][key_].append(err_ml)
                err_dd_dic['pw_zero'][key_].append(err_dd)


                ## distribution
                data,size_factor = dl.poi_data_gen_nd(p_true,x_true,sigma=0,Nc=Nc,Nr=Nr,random_seed=i_rep)
                err_ml,err_dd = 0,0
                for i_dim in range(data.shape[1]):
                    Y = np.array(data[:,i_dim].X).astype(int)

                    p_ml = np.bincount(Y)
                    p_ml = p_ml / np.sum(p_ml)
                    x_ml = np.arange(p_ml.shape[0])

                    p_dd,x_dd = sd.dd_distribution(Y,gamma=None,verbose=False)

                    p_true_,x_true_ = dist_true[i_dim]

                    err_ml += dist_W1(p_true_,p_ml,x_true_,x_ml)/data.shape[1]
                    err_dd += dist_W1(p_true_,p_dd,x_true_,x_dd)/data.shape[1]


                err_ml_dic['dist'][key_].append(np.log10(err_ml))
                err_dd_dic['dist'][key_].append(np.log10(err_dd))

    ## Save the data
    fname = output_folder+'/data.pickle'
    f_myfile = open(fname,'wb')
    pickle.dump(err_ml_dic, f_myfile)
    pickle.dump(err_dd_dic, f_myfile)
    f_myfile.close()

    logger.info('total time: %0.1f'%(time.time()-start_time))
    
if __name__ == '__main__':
    main()
