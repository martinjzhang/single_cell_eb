import numpy as np
import scipy as sp
import scipy.sparse as sp_sparse
import matplotlib.pyplot as plt
import h5py
from util import *

## data loading
def load_Zeisel():
    X_label={}
    fil_Zeisel='/home/martin/data/single_cell/Zeisel/expression_mRNA_17-Aug-2014.txt'
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

def load_10x():
    X_label={}
    X=None
    fil_path='/home/martin/data/single_cell/10x_1.3mil_mice_brain/1M_neurons_filtered_gene_bc_matrices_h5.h5'
    fil_cluster_path='/home/martin/single_cell_eb/data/10x_1.3mil_mice_brain/analysis/clustering/graphclust/clusters.csv'
    
    ## load the data matrix and the gene names
    f=h5py.File(fil_path)
    dsets=f[list(f.keys())[0]]
    X=sp_sparse.csc_matrix((dsets['data'][()], dsets['indices'][()], dsets['indptr'][()]), shape=dsets['shape'][()])
    X=(X.T).tocsc()
    gene_name=dsets['gene_names'][()]
    
    ## load the cluster labels
    barcodes=dsets['barcodes'][()]
    X_label=np.zeros(len(barcodes),dtype=int)
    ct=0
    with open(fil_cluster_path) as f:
        for line in f:
            line=line.strip().split(',')
            if ct>0:
                X_label[ct-1]=int(line[1])
            ct+=1
    data_summary(X,X_label,gene_name)           
    return X,X_label,gene_name

def data_summary(X,X_label,gene_name):
    print('###### Summary ######')
    print('GC matrix: ',X.shape)
    print('number of genes:', len(gene_name))
    #print 'number of clusters:', np.unique(X_label).shape[0]
    print('###### End Summary ######')