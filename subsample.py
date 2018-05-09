import logging
import numpy as np
import scanpy.api as sc
import scdd as sd

from data_loader import * 


def main():
    output_dir = '/data/martin/exp_sceb/subsample_1.3mil'
    ## have a logger 
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=output_dir+'/subsample_info.log', filemode='w')
      
    ## actual code 
    data = load_10x_1_3mil_subsample()
    logging.info('Data loaded: n_cell=%d, n_gene=%d'%(data.shape[0],data.shape[1]))
    
    ## calculate the parameters 
    Nc,G = data.shape
    Nr = data.X.sum()/Nc
    B = Nc*Nr
    Nr_bar = Nr/G
    logging.info('Budget = %d, G = %d, Nc = %d, Nr = %0.2f, Nr_bar = %0.2f'%(B,G,Nc,Nr,Nr_bar))
        
    ## assigning the parameters 
    B_sub_list = [int(B/10000),int(B/5000),int(B/2000),int(B/1000)]
    param=[]    # a tuple of (B,Nr_bar,Nr,Nc)
    rep_time = 100
    for B_sub in B_sub_list:
        for Nr_bar in [0.5,1,2,3,4,5,8,10]:
            for i_rep in range(rep_time): 
                param.append((B_sub,Nr_bar,Nr_bar*G,int(B_sub/Nr_bar/G),i_rep))
                
    ## subsampling 
    output_folder = '/data/martin/exp_sceb/subsample_1.3mil/subsampled_data'
    for i in range(len(param)):
    #for i in range(2):
        B_sub,Nr_bar_,Nr_,Nc_,i_rep = param[i]
        temp_str = '%d_%0.1f_%0.1f_%d_%d'%(param[i])
        logging.info('Writing: '+temp_str)
        temp = sd.subsample_anndata(data,Nr_,Nc_,random_state=i_rep,verbose=False)
        temp.write(filename=output_folder+'/data_'+temp_str+'.h5ad')

if __name__ == '__main__':
    main()
