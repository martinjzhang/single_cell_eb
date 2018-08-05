import logging
import os
from sys import argv

dname = argv[1]

output_folder = '/home/martin/single_cell_eb/figures/figure_efficiency'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logging.basicConfig(level=logging.INFO,\
        format='%(module)s:: %(message)s',\
        filename=output_folder+'/result_%s.log'%dname, filemode='w')
logger = logging.getLogger(__name__)

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

def main():   
    start_time = time.time()
    if dname == 'pbmc_4k':
        data = dl.load_10x_4k() # 4k PBMC cells
    elif dname == 'brain_9k':
        data = dl.load_10x_9k()
    elif dname == 'brain_1.3m':
        data = dl.load_10x_1_3mil()
    logger.info('# load data: %0.4fs\n'%(time.time()-start_time))
    
    data.var_names_make_unique()        
    mean_count = np.array(data.X.mean(axis=0)).reshape(-1)
    gene_list = np.array(list(data.var_names))
    gene_list = gene_list[np.argsort(mean_count)[::-1][0:4000]]
    data = data[:,list(gene_list)]    
    
    #sc.pp.filter_genes(data,min_counts=0.1*data.shape[0])
    size_factor = sd.dd_size_factor(data,verbose=False)
    _ = sd.get_info(data,logger=logger)
    
    n_rep = 5
    
    # do it one time first 
    _ = sd.dd_1d_moment(data,size_factor=size_factor,verbose=False)
    
    ## dd_1d_moment
    time_dd_1d_moment = np.zeros([n_rep])
    for i in range(n_rep):      
        start_time = time.time()
        _ = sd.dd_1d_moment(data,size_factor=size_factor,verbose=False)
        time_dd_1d_moment[i] = time.time() - start_time
        logger.info('# dd_1d_moment, i_rep=%d, time=%0.4f'%(i,time_dd_1d_moment[i]))        
    logger.info('# avg time=%0.4fs, std=%0.4f\n'%(time_dd_1d_moment.mean(),time_dd_1d_moment.std()))
    
    ## dd_covariance
    time_dd_covariance = np.zeros([n_rep])
    for i in range(n_rep):      
        start_time = time.time()
        _ = sd.dd_covariance(data,size_factor=size_factor,verbose=False)
        time_dd_covariance[i] = time.time() - start_time
        logger.info('# dd_covariance, i_rep=%d, time=%0.4f'%(i,time_dd_covariance[i]))        
    logger.info('# avg time=%0.4fs, std=%0.4f\n'%(time_dd_covariance.mean(),time_dd_covariance.std()))
    
    ## dd_inactive_prob
    time_dd_inactive_prob = np.zeros([n_rep])
    for i in range(n_rep):      
        start_time = time.time()
        _ = sd.dd_inactive_prob(data,relative_depth=1,\
                                size_factor=size_factor,verbose=False)
        time_dd_inactive_prob[i] = time.time() - start_time
        logger.info('# dd_inactive_prob, i_rep=%d, time=%0.4f'%(i,time_dd_inactive_prob[i]))        
    logger.info('# avg time=%0.4fs, std=%0.4f\n'%(time_dd_inactive_prob.mean(),time_dd_inactive_prob.std()))
    
    ## dd_pairwise_inactive_prob
    time_dd_pairwise_inactive_prob = np.zeros([n_rep])
    for i in range(n_rep):      
        start_time = time.time()
        _ = sd.dd_pairwise_inactive_prob(data,relative_depth=1,size_factor=size_factor,verbose=False)
        time_dd_pairwise_inactive_prob[i] = time.time() - start_time
        logger.info('# time_dd_pairwise_inactive_prob, i_rep=%d, time=%0.4f'%(i,time_dd_pairwise_inactive_prob[i]))        
    logger.info('# avg time=%0.4fs, std=%0.4f\n'%(time_dd_pairwise_inactive_prob.mean(),time_dd_pairwise_inactive_prob.std()))
    
if __name__ == '__main__':
    main()
