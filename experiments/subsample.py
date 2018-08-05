import logging
import numpy as np
import os
import pickle

import scanpy.api as sc
import sceb.data_loader as dl
import sceb.scdd as sd

def main():
    opt = 'full'
    #opt = 'small'
    output_dir = '/data/martin/exp_sceb/subsample_1.3mil'

    ## have a logger 
    if opt == 'full':
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            filename=output_dir+'/subsample_info.log', filemode='w')
    else:
        logging.basicConfig(level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s',
            filename=output_dir+'/subsample_small_info.log', filemode='w')
      
    ## actual code 
    data = dl.load_10x_1_3mil_subsample(opt=10)
    logging.info('Data loaded: n_cell=%d, n_gene=%d'%(data.shape[0],data.shape[1]))
    
    ## calculate the parameters 
    Nc,G = data.shape
    Nr = data.X.sum()/Nc
    B = Nc*Nr
    Nr_bar = Nr/G
    logging.info('Budget = %d, G = %d, Nc = %d, Nr = %0.2f, Nr_bar = %0.2f'%(B,G,Nc,Nr,Nr_bar))
        
        
    ## Read subsample parameters
    fname = '/home/martin/single_cell_eb/figures/figure_tradeoff_curve/subsample_param.pickle'
    f_myfile = open(fname, 'rb')
    B_sub_list = pickle.load(f_myfile)
    subsample_param_dic = pickle.load(f_myfile)
    f_myfile.close()

    ## assigning the parameters 
    
    #B_sub_list = [int(B/15000), int(B/4000),int(B/1000),int(B/250)]
    param=[]    # a tuple of (B,Nr_bar,Nr,Nc)
    
    if opt == 'full':
        rep_time = 100
    else:
        rep_time = 2
                
    for B_sub in subsample_param_dic.keys():
        for i in range(len(subsample_param_dic[B_sub][0])):
            for i_rep in range(rep_time): 
                Nr_bar_sub = subsample_param_dic[B_sub][0][i]
                Nc_sub = subsample_param_dic[B_sub][1][i]
                Nr_sub = Nr_bar_sub*G
                if Nc_sub < Nc:
                    param.append((B_sub,Nr_bar_sub,Nr_sub,Nc_sub,i_rep))
            #key_ = '%d_%0.1f_%0.1f_%d'%(B,Nr_bar,Nr,Nc)
            
    for i in range(len(param)):
        print(param[i])
                
    ## subsampling 
    if opt == 'full':
        output_folder = output_dir+'/subsampled_data_new'
    else: 
        output_folder = output_dir+'/subsampled_data_small_new'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for i in range(len(param)):
    #for i in range(2):
        B_sub,Nr_bar_,Nr_,Nc_,i_rep = param[i]
        temp_str = '%d_%0.2f_%0.2f_%d_%d'%(param[i])
        logging.info('Writing: '+temp_str)
        temp = sd.subsample_anndata(data,Nr_,Nc_,random_state=i_rep,verbose=False)
        temp.write(filename=output_folder+'/data_'+temp_str+'.h5ad')

if __name__ == '__main__':
    main()
