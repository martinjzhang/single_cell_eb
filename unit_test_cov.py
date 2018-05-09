import matplotlib
matplotlib.use('Agg')
import logging
import os
import time
import matplotlib.pyplot as plt

import data_loader as dl
import scdd as sd

def main():
    ## create a new directory
    output_folder = './results/unit_test_cov'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    ## have a logger 
    logging.basicConfig(level=logging.DEBUG,
        format='%(module)s:: %(message)s',
        filename=output_folder+'/result.log', filemode='w')
    logger = logging.getLogger()
    
    ## get the data 
    data,X,p0_true,size_factor = load_toy_ann_data()
    cov_true = np.cov(X.T)
    logger.info('## True covariance matrix')
    logger.info(str(cov_true))

    ## estimation 
    
if __name__ == '__main__':
    main()