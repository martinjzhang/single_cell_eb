## First level of configuration
import logging
import os
output_folder = './results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logging.basicConfig(level=logging.INFO,\
        format='%(module)s:: %(message)s',\
        filename=output_folder+'/result_1d_moment.log', filemode='w')
logger = logging.getLogger(__name__)

## Other packages
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import numpy as np
import sceb.data_loader as dl
import sceb.scdd as sd

## fix it: add the test for zero component 
def main():    
    ## get the data 
    data,X,size_factor = dl.load_toy_ann_data()
    Nc,G,Nr,Nr_bar = sd.get_info(data)
    logger.info('## True moments')
    for i in range(4):
        logger.info('%d-th moment: '%(i+1)+str(np.mean(X**(i+1),axis=0)))
    logger.info('')

    ## estimation: dd
    start_time = time.time()
    logger.info('######## unit test: dd_1d_moment ########')
    start_time = time.time()
    M_ml,M_dd  = sd.dd_1d_moment(data,size_factor=size_factor,verbose=True,k=4,Nr=Nr)
    logger.info('## M_ml ')
    for i in range(M_ml.shape[0]):
        logger.info(str(M_ml[i,:]))
    
    logger.info('')
    logger.info('## M_dd ')
    for i in range(M_dd.shape[0]):
        logger.info(str(M_dd[i,:]))
    
    logger.info('')
    logger.info('## total time: %0.2fs\n'%(time.time()-start_time))    
              
if __name__ == '__main__':
    main()