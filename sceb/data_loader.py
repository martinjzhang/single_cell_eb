import numpy as np
import scipy as sp
import pandas as pd
import scipy.sparse as sp_sparse
import matplotlib.pyplot as plt
import h5py
import sceb.b_spline_nd as spl
from sceb.util import *
import scanpy.api as sc

"""
    a 1d toy count data, for simulation of distribution estimation
"""
def load_1d_toy_spline(verbose=False):
    #alpha = np.array([1,2.4,2.15,1.2,-0.7 ,-5])
    alpha = np.array([3,2,-4,5,-2,-2])
    #alpha = np.array([3,2.4,2.15,1.2,-0.7,-5])
    x_grid     = np.linspace(0,1,101)
    Q,_   = spl.Q_gen_ND(points=x_grid,n_degree=5,zero_inflate=True,verbose=verbose)
    p     = np.exp(Q.dot(alpha))
    p    /= np.sum(p)
    if verbose:
        print(alpha)
    
    if verbose:
        plt.figure()
        plot_density_1d(p,x_grid)
        plt.xlabel('support')
        plt.ylabel('probablity')
        plt.title('toy distribution, mean: %s'%str(np.sum(p*x_grid)))
        plt.legend()
        plt.show()   
    return p,x_grid  

def poi_data_gen(p,x_grid,Nc=10000,Nr=5,G=2,require_X=False,sigma=0.2):
    
    X = np.zeros([Nc,G],dtype=float)
    for i in range(G):
        temp = np.random.choice(x_grid,Nc,p=p,replace=True)  
        X[:,i] = temp
    #X[:,-1] = 1
    #X = (X.T/np.sum(X,axis=1)).T    # normalize to be a probability distribution    
    new_Nr = Nr*Nc/X.sum()
    
    ## sample the size factor
    size_factor = np.random.randn(Nc)*sigma + 1 
    size_factor = size_factor.clip(min=0.5)
    
    ## generating the reads 
    Y = np.random.poisson((X.T*size_factor).T*new_Nr)
    Y = sp.sparse.csr_matrix(Y)
    
    ## assign some fake gene names
    gene_name = []
    for i in range(G):
        gene_name.append('gene %d'%(i))        
    var = pd.DataFrame(index=gene_name)
    
    data = sc.AnnData(Y,var=var)
    
    if require_X:
        return data,size_factor,X
    else:
        return data,size_factor
    
def poi_data_gen_nd(p,val,Nc=10000,Nr=5,sigma=0.2,random_seed=0):
    np.random.seed(910624+random_seed)
    val_size,G = val.shape
    rand_ind = np.random.choice(np.arange(val_size),Nc,p=p, replace=True)    
    X= val[rand_ind,:] 
    
    new_Nr = Nr*Nc/X.sum()
    
    ## sample the size factor
    size_factor = np.random.randn(Nc)*sigma + 1 
    #size_factor = np.random.randn(Nc)*0 + 1 
    size_factor = size_factor.clip(min=0.5)
    
    ## generating the reads 
    Y = np.random.poisson((X.T*size_factor).T*new_Nr)
    Y = sp.sparse.csr_matrix(Y)
    
    ## assign some fake gene names
    gene_name = []
    for i in range(G):
        gene_name.append('gene %d'%(i))        
    var = pd.DataFrame(index=gene_name)
    
    data = sc.AnnData(Y,var=var)
    
    return data,size_factor

"""
    a toy ann data, for testing the algorithms
"""

# fixit: incorporate dependency on the data 
def load_toy_ann_data(verbose=False,Nc=100000,Nr=5,logger=None):
    np.random.seed(42)
    G = 3
    
    ## sample the gene expression level 
    p = np.array([0.2,0.1,0.2,0.1,0.1,0.2,0.1],dtype=float)
    x = np.array([[0,0],[0,0.1],[0.1,0],[0.1,0.1],[0,0.2],[0.2,0],[0.2,0.2]],dtype=float)
    
    X = np.zeros([Nc,G],dtype=float)
    temp = np.random.choice(np.arange(x.shape[0]),Nc, p=p,replace=True)  
    X[:,0:2] = x[temp]
    X[:,-1] = 0.1
    X = (X.T/np.sum(X,axis=1)).T    # normalize to be a probability distribution    

    ## sample the size factor
    size_factor = np.random.randn(Nc)*0.2 + 1 
    #size_factor = np.random.randn(Nc)*0 + 1 
    size_factor = size_factor.clip(min=0.5)
    
    ## generating the reads 
    Y = np.random.poisson((X.T*size_factor).T*Nr)
    Y = sp.sparse.csr_matrix(Y)
    
    ## assign some fake gene names
    gene_name = []
    for i in range(G):
        gene_name.append('gene %d'%(i))        
    var = pd.DataFrame(index=gene_name)
    
    data = sc.AnnData(Y,var=var)
    
    return data,X,size_factor

"""
    Zeisel data
"""
def load_Zeisel():
    X_label={}
    fil_Zeisel='/data/martin/data/single_cell/Zeisel/expression_mRNA_17-Aug-2014.txt'
    f=open(fil_Zeisel,'rU')
    X=[]
    gene_name=[]
    ct=0
    for line in f:
        line=line.strip().split('\t')
        if ct <=9:
            print(ct,line[0:3],line[-2:],len(line))    
            if 'none' not in line[0]:
                if is_number(line[1]) is True:
                    X_label[line[0]]=np.array(line[1:],dtype=float) 
                else:
                    X_label[line[0]]=np.array(line[1:]) 
            else:
                if is_number(line[1]) is True:
                    X_label[line[1]]=np.array(line[2:],dtype=float)
                else:
                    X_label[line[1]]=np.array(line[2:])
        if ct>10:
            gene_name.append(line[0])
            X.append(line[2:])
        ct+=1
    f.close()
    X=np.array(X,dtype=float).T
    for i in X_label.keys():
        print(i,len(X_label[i]),X_label[i][0:5])
    gene_name=np.array(gene_name)
    data_summary(X,X_label,gene_name)
    return X,X_label,gene_name

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.3.0/1M_neurons
"""
def load_10x_1_3mil():
    filename_data='/data/martin/single_cell/10x_1.3mil_mice_brain/1M_neurons_filtered_gene_bc_matrices_h5.h5'
    data=sc.read_10x_h5(filename_data)
    return data

def load_10x_1_3mil_subsample(opt=10):
    if opt==10:
        filename_data='/data/martin/exp_sceb/subsample_1.3mil/data_1.3mil_high10_gene.h5ad'
    elif opt==5:
        filename_data='/data/martin/exp_sceb/subsample_1.3mil/data_1.3mil_high5_gene.h5ad'
    elif opt==1:
        filename_data='/data/martin/exp_sceb/subsample_1.3mil/data_1.3mil_high1_gene.h5ad'
    elif opt==0.5:
        filename_data='/data/martin/exp_sceb/subsample_1.3mil/data_1.3mil_high0.5_gene.h5ad'
    data=sc.read(filename_data)
    return data 

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neurons_2000
"""
def load_10x_2k():
    # 2k mice brain @ 100K rpc
    filename_data = '/data/martin/single_cell/10x_2k_brain_cell/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_2k_brain_cell/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_2k_brain_cell/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neuron_9k
"""
def load_10x_9k():
    filename_data = '/data/martin/single_cell/10x_9k_brain_cell_e18_mouse/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_9k_brain_cell_e18_mouse/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_9k_brain_cell_e18_mouse/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neurons_900
"""
def load_10x_1k():
    filename_data = '/data/martin/single_cell/10x_1k_brain_e18_mouse/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_1k_brain_e18_mouse/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_1k_brain_e18_mouse/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data


"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/fresh_68k_pbmc_donor_a
"""
def load_10x_68k():
    filename_data = '/data/martin/single_cell/10x_68k_PBMC/filtered_matrices_mex/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_68k_PBMC/filtered_matrices_mex/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_68k_PBMC/filtered_matrices_mex/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc33k
"""
def load_10x_33k():
    filename_data = '/data/martin/single_cell/10x_33k_pbmc/filtered_gene_bc_matrices/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_33k_pbmc/filtered_gene_bc_matrices/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_33k_pbmc/filtered_gene_bc_matrices/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc3k
"""
def load_10x_3k():
    filename_data = '/data/martin/single_cell/10x_3k_pbmc/filtered_gene_bc_matrices/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_3k_pbmc/filtered_gene_bc_matrices/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_3k_pbmc/filtered_gene_bc_matrices/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

"""
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc6k
"""
def load_10x_6k():
    filename_data = '/data/martin/single_cell/10x_pbmc_6k/filtered_matrices_mex/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_pbmc_6k/filtered_matrices_mex/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_pbmc_6k/filtered_matrices_mex/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc8k
"""
def load_10x_8k():
    filename_data = '/data/martin/single_cell/10x_8k_PBMC/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_8k_PBMC/filtered_gene_bc_matrices/GRCh38/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_8k_PBMC/filtered_gene_bc_matrices/GRCh38/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc4k
"""
def load_10x_4k():
    filename_data = '/data/martin/single_cell/10x_4k_PBMC/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_4k_PBMC/filtered_gene_bc_matrices/GRCh38/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_4k_PBMC/filtered_gene_bc_matrices/GRCh38/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/t_3k
"""
def load_10x_3k_panT():
    filename_data = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_3k_panT/filtered_gene_bc_matrices/GRCh38/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/t_4k
"""
def load_10x_4k_panT():
    filename_data = '/data/martin/single_cell/10x_4k_panT/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_4k_panT/filtered_gene_bc_matrices/GRCh38/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_4k_panT/filtered_gene_bc_matrices/GRCh38/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_1k
"""
def load_10x_1k_mix_human():
    filename_data = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

def load_10x_1k_mix_mouse():
    filename_data = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_1k_mix/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_6k
"""
def load_10x_6k_mix_human():
    filename_data = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

def load_10x_6k_mix_mouse():
    filename_data = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_6k_mix/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

""" 
    https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/hgmm_12k
"""
def load_10x_12k_mix_human():
    filename_data = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/hg19/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/hg19/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/hg19/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

def load_10x_12k_mix_mouse():
    filename_data = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/mm10/matrix.mtx'
    filename_genes = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/mm10/genes.tsv'
    filename_barcodes = '/data/martin/single_cell/10x_12k_mix/filtered_gene_bc_matrices/mm10/barcodes.tsv'

    data = sc.read(filename_data, cache=True).transpose()
    data.var_names = np.genfromtxt(filename_genes, dtype=str)[:, 1]
    data.smp_names = np.genfromtxt(filename_barcodes, dtype=str)
    return data

def data_summary(X,X_label,gene_name):
    print('###### Summary ######')
    print('GC matrix: ',X.shape)
    print('number of genes:', len(gene_name))
    #print 'number of clusters:', np.unique(X_label).shape[0]
    print('###### End Summary ######')