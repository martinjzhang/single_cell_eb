## First level of configuration
import logging
import os
output_folder = './results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
logging.basicConfig(level=logging.INFO,\
        format='%(module)s:: %(message)s',\
        filename=output_folder+'/result_dd_inactive_prob.log', filemode='w')
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
    kappa = 5
    for Nr in [2,4,10]:
        data,X,size_factor = dl.load_toy_ann_data(Nr=Nr)
        Nc,G,Nr,Nr_bar = sd.get_info(data)
        
        logger.info('## Nr=%0.4f, kappa=%d'%(Nr,kappa))
        logger.info('## True dd_inactive_prob')
        logger.info('# '+str(np.mean(np.exp(-kappa*X),axis=0)))
        logger.info('')


        ## estimation: dd
        start_time = time.time()
        logger.info('######## unit test: dd_inactive_prob ########')
        start_time = time.time()
        p0_ml,p0_dd  = sd.dd_inactive_prob(data,size_factor=size_factor,relative_depth=Nr/kappa,verbose=False)
        logger.info('## p0_ml ')
        logger.info(str(p0_ml))

        logger.info('')
        logger.info('## p0_dd ')
        logger.info(str(p0_dd))

        logger.info('')
        logger.info('## total time: %0.2fs\n'%(time.time()-start_time))    
        logger.info('')
              
if __name__ == '__main__':
    main()