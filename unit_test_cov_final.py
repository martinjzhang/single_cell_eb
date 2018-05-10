## First level of configuration
import logging
import os
output_folder = './results/unit_test_cov'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logging.basicConfig(level=logging.INFO,\
        format='%(module)s:: %(message)s',\
        filename=output_folder+'/result.log', filemode='w')
logger = logging.getLogger(__name__)

## Other packages
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import numpy as np
import data_loader as dl
import scdd as sd

def main():    
    ## get the data 
    data,X,p0_true,size_factor = dl.load_toy_ann_data()
    logger.info('## True mean')
    logger.info(str(X.mean(axis=0)))
    cov_true = np.cov(X.T)
    logger.info('## True covariance matrix')
    for i in range(cov_true.shape[0]):
        logger.info(str(cov_true[i,:]))
    PC_true = np.corrcoef(X.T)
    logger.info('## True Pearson correlation')
    for i in range(PC_true.shape[0]):
        logger.info(str(PC_true[i,:]))
    logger.info('\n')

    ## estimation: dd
    logger.info('######## unit test: dd_covariance ########')
    start_time = time.time()
    mean_dd,cov_dd,PC_dd  = sd.dd_covariance(data,size_factor=size_factor)
    logger.info('## dd mean ')
    logger.info(str(mean_dd))
    logger.info('')
    logger.info('## dd covariance matrix')
    for i in range(cov_dd.shape[0]):
        logger.info(str(cov_dd[i,:]))
    logger.info('')
    logger.info('## dd Pearson correlation')
    for i in range(PC_dd.shape[0]):
        logger.info(str(PC_dd[i,:]))
    logger.info('')
    logger.info('## total time: %0.2fs\n'%(time.time()-start_time))    
        
    ## estimation: ml
    logger.info('######## unit test: ml_covariance ########')
    start_time = time.time()
    mean_ml,cov_ml,PC_ml  = sd.ml_covariance(data,size_factor=size_factor)
    logger.info('## ml mean ')
    logger.info(str(mean_ml))
    logger.info('')
    logger.info('## ml covariance matrix')
    for i in range(cov_ml.shape[0]):
        logger.info(str(cov_ml[i,:]))
    logger.info('')
    logger.info('## ml Pearson correlation')
    for i in range(PC_ml.shape[0]):
        logger.info(str(PC_ml[i,:]))
    logger.info('')
    logger.info('## total time: %0.2fs\n'%(time.time()-start_time))  
    
    logger.info('######## unit test: ml brute force (comparison) ########')
    Y = data.X.todense()
    Nr = data.X.sum()/data.shape[0]
    Y_sf = (Y.T/size_factor).T
    logger.info('## ml_bf mean ')
    logger.info(str(Y_sf.mean(axis=0)/Nr))
    logger.info('')
    logger.info('## ml_bf covariance matrix')
    cov_ml = np.cov(Y_sf.T)/Nr**2
    for i in range(cov_ml.shape[0]):
        logger.info(str(cov_ml[i,:]))
    logger.info('')
    logger.info('## ml_bf Pearson correlation')
    PC_ml = np.corrcoef(Y_sf.T)
    for i in range(PC_ml.shape[0]):
        logger.info(str(PC_ml[i,:]))        
    
if __name__ == '__main__':
    main()