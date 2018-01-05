import numpy as np
import scipy as sp
import scipy.sparse as sp_sparse
import matplotlib.pyplot as plt
import h5py

## data loading
def load_Zeisel():
    X_label={}
    fil_Zeisel='/home/martin/single_cell_eb/data/Zeisel/expression_mRNA_17-Aug-2014.txt'
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
    return X,X_label,gene_name

def load_10x():
    X_label={}
    X=None
    fil_path='/home/martin/single_cell_eb/data/10x_1.3mil_mice_brain/1M_neurons_filtered_gene_bc_matrices_h5.h5'
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
    return X,X_label,gene_name

def pre_process(X,gene_name):
    ## filter 
    feature_select=np.ones(gene_name.shape,dtype=bool)
    temp=np.sum(X,axis=0)
    feature_select[temp<50]=0
    
    ## select the features
    X=np.log(X[:,feature_select]+1)
    gene_name=gene_name[feature_select]
    
    return X,gene_name
    

def plot_scatter(X,X_label,idx1,idx2):
    #plt.figure()
    for i in np.unique(X_label):
        plt.scatter(X[X_label==i,idx1],X[X_label==i,idx2],alpha=0.4,s=10,label=str(i))
    #plt.legend()
    #plt.show()
    
def plot_hist(X,X_label=None,n_bin=100):
    bins_=np.linspace(np.min(X),np.max(X),n_bin)
    #plt.figure()
    if X_label is None:
        plt.hist(X,alpha=0.4,bins=bins_)
    else:
        for i in np.unique(X_label):
            plt.hist(X[X_label==i],alpha=0.4,bins=bins_,label=str(i))
    #plt.legend()
    #plt.show()
def plot_gene(X,gene_name,X_label=None,n_bin=100):
    plt.figure(figsize=[10,5])
    plt.subplot(1,2,1)
    plt.title(gene_name)
    plot_hist(X,n_bin=n_bin)
    plt.subplot(1,2,2)
    plot_hist(X,X_label,n_bin=n_bin)
    plt.show()

def plot_density_1d(p,x):
    plt.plot(x,p,marker='o')

def plot_density_2d(p,x):
    plt.scatter(x[:,0], x[:,1],s=5000*p,alpha=0.8,c=p,cmap='viridis')
    plt.colorbar()
    plt.title('PC: %s,  MI: %s'%(str(pearson_corr(p,x))[0:6],str(mutual_info(p,x))[0:6]))
    
def plot_pair(X,X_label,idx1,idx2):
    plt.figure(figsize=[20,7.5])
    plt.subplot(1,3,1)
    plot_scatter(X,X_label,idx1,idx2)
    plt.title('scatter plot of gene %s vs gene %s'%(str(idx1),str(idx2)))
    plt.subplot(1,3,2)
    plot_hist(X[:,idx1],X_label)
    plt.title('histogram of gene %s'%str(idx1))
    plt.subplot(1,3,3)
    plot_hist(X[:,idx2],X_label)
    plt.title('histogram of gene %s'%str(idx2))
    plt.show()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

    
def pearson_corr(p,x):
    n_supp = x.shape[0]
    mean1  = 0.0
    mean2  = 0.0
    var1   = 0.0
    var2   = 0.0
    cov    = 0.0
    
    for i in range(n_supp):
        mean1 += p[i]*x[i,0]
        mean2 += p[i]*x[i,1]
        var1  += p[i]*(x[i,0]**2)
        var2  += p[i]*(x[i,1]**2)
        cov   += p[i]*x[i,0]*x[i,1]
    
    cov  -= mean1*mean2
    var1 -= mean1**2
    var2 -= mean2**2
    
    if var1*var2>0:
        return cov/np.sqrt(var1*var2)
    else:
        return 0
    
def mutual_info(p,x):
    n_supp = x.shape[0]
    p1     = {}
    p2     = {}
    mi     = 0.0
    
    ## compute the marginals 
    for i in range(n_supp):
        if x[i,0] not in p1.keys():
            p1[x[i,0]] = p[i]
        else:
            p1[x[i,0]] += p[i]
        if x[i,1] not in p2.keys():
            p2[x[i,1]] = p[i]
        else:
            p2[x[i,1]] += p[i]
    
    ## compute the mutual information
    for i in range(n_supp):
        if p[i]>0:
            mi += p[i]*np.log(p[i]/(p1[x[i,0]]*p2[x[i,1]]))
            
    return mi

def info_2d(p,x):
    return pearson_corr(p,x),mutual_info(p,x)

def dist_kl(p1,p2):
    kl = 0.0
    for i in range(p1.shape[0]):
        if p1[i] > 0:
            kl+= p1[i] * np.log(p2[i]/p1[i])
    return kl

def dist_tv(p1,p2):
    return 0.5*np.linalg.norm(p1-p2,ord=1)

def quantize(p,x,n_bin=10):
    x_q = []
  
    for i in range(n_bin+1):
        for j in range(n_bin+1):
            x_q.append([i,j])
    x_q = np.array(x_q,dtype=float)
    x_q = x_q/(n_bin+0.0)
    p_q = np.zeros([x_q.shape[0]],dtype=float)
    
    for i in range(x.shape[0]):
        j = np.argmin(np.linalg.norm(x_q-x[i,:],ord=1,axis=1))
        p_q[j] += p[i]
        
    return p_q,x_q
    

    
    