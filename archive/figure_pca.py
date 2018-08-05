## First level of configuration
import logging
import os
output_folder = './figures/figure_pca'
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
import scanpy.api as sc

def plot_PCA_score(data,v1,v2,label=None,n_cell=None,ref_score=None,return_score=False):
    if n_cell is not None:
        subsample_index = np.random.choice(np.arange(data.shape[0]),n_cell,replace=False)  
    X=data.X
    score1 = X.dot(v1)
    score2 = X.dot(v2)
    
    if ref_score is not None:
        ref_score1 = ref_score[0]
        ref_score2 = ref_score[1]
        
        ref_score1 /= ref_score1.std()
        ref_score2 /= ref_score1.std()
        
        if np.linalg.norm(ref_score1-score1/score1.std())>np.linalg.norm(ref_score1+score1/score1.std()):
            score1 = -score1
        if np.linalg.norm(ref_score2-score2/score2.std())>np.linalg.norm(ref_score2+score2/score2.std()):
            score2 = -score2
    
    plt.scatter(score1,score2,s=4,alpha=0.3)
    if return_score:
        return [score1,score2]
    
def preprocess(data):
    data_ = data.copy()
    A = data_.X.data
    cap = np.percentile(A,99)
    A = A.clip(max=cap)
    data_.X.data = A
    return data_

def main():    
    
    sub_sample_rate = [1.5,1.8,2.1,2.5,3,4,5,8,10]
    #sub_sample_rate = [2.5]
    gene_filter_rate = [2,1.5,1,0.5,0.2,0.1]
    #gene_filter_rate = [5]
    
    for sr in sub_sample_rate:
        for gr in gene_filter_rate:
    
            ## get the data 
            data_raw = dl.load_10x_4k()
            data_name = 'pbmc_4k'
            sc.pp.filter_genes(data_raw,min_counts=gr*data_raw.shape[0])    
            logger.info('## load %s data'%data_name)
            Nc,G = data_raw.shape
            Nr = data_raw.X.sum()/Nc
            Nr_sub = Nr/sr
            logger.info('## Nc=%d, G=%d, Nr=%0.1f, Nr_sub=%0.1f'%(Nc,G,Nr,Nr_sub))    

            ## subsample the data
            data_subsample_raw = sd.subsample_anndata(data_raw,Nr_sub,Nc)

            ## pca on the original data
            data_full = preprocess(data_raw)
            mean_full,cov_full,PC_full  = sd.ml_covariance(data_full,size_factor=None)
            D_full,V_full = np.linalg.eigh(cov_full)

            ## pca on the subsampled data: ml,dd
            data_subsample = preprocess(data_subsample_raw)
            mean_ml,cov_ml,PC_ml  = sd.ml_covariance(data_subsample,size_factor=None)
            D_ml,V_ml = np.linalg.eigh(cov_ml)

            mean_dd,cov_dd,PC_dd  = sd.dd_covariance(data_subsample,size_factor=None)
            D_dd,V_dd = np.linalg.eigh(cov_dd)

            ## Visualization
            v_list = [[1,2],[2,3],[4,5],[6,7]]
            plt.figure(figsize=[18,24])
            for i_fig,i_PC in enumerate(v_list):
                i_v1, i_v2 = i_PC
                plt.subplot(4,3,i_fig*3+1)
                ref_score = plot_PCA_score(data_raw,V_full[:,-i_v1],V_full[:,-i_v2])
                plt.xlabel('PC %d'%i_v1)
                plt.ylabel('PC %d'%i_v2)
                plt.title('Full data with plug-in estimator')
                plt.subplot(4,3,i_fig*3+2)
                plot_PCA_score(data_subsample_raw,V_ml[:,-i_v1],V_ml[:,-i_v2],ref_score=ref_score)
                plt.title('Subsampled data with plug-in estimator')
                plt.xlabel('PC %d'%i_v1)
                plt.ylabel('PC %d'%i_v2)
                plt.subplot(4,3,i_fig*3+3)
                plot_PCA_score(data_subsample_raw,V_dd[:,-i_v1],V_dd[:,-i_v2],ref_score=ref_score)
                plt.title('Subsampled data with EB estimator')
                plt.xlabel('PC %d'%i_v1)
                plt.ylabel('PC %d'%i_v2)
            plt.savefig(output_folder+'/pca_scatter_'+data_name+'_ngene_%d_Nrsub_%0.1f.png'%(G,Nr_sub))
            plt.close('all')
   
    
if __name__ == '__main__':
    main()