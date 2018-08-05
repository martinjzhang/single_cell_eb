import numpy as np
import scipy as sp
import scipy.stats
from scipy.stats import wasserstein_distance
import scipy.sparse as sp_sparse
import matplotlib.pyplot as plt
import h5py

'''
    calculating the size factor (not scalable so far ...) 
'''
def size_factor(X,verbose=False):
    sf = X.mean(axis=1)/X.mean()
    sf = np.array(sf).flatten()
    if verbose:
        plt.figure(figsize=[12,5])
        plt.hist(sf,bins=100)
        plt.show()
    return sf

def sub_sample(Y,n_sub):
    n_cell = Y.shape[0]
    sample = np.zeros(int(np.sum(Y)),dtype=int)
    loc = 0
    for i in range(n_cell):
        sample[loc:loc+Y[i]] = i
        loc += Y[i]
    sample = sample[np.random.permutation(sample.shape[0])[0:n_sub]]
    Y[:] = 0
    for i in sample:
        Y[i] += 1
    return Y

## some ancillary functions
"""
    calculate the dimension-wise rank statistics
"""
def rank(x,opt='ascent'):
    ranks = np.empty_like(x)
    temp = x.argsort(axis=0)
    ranks[temp] = np.arange(x.shape[0])
    if opt=='descent':
        ranks=x.shape[0]-1-ranks
    return ranks

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def entropy(p,x):
    p = p[p>0]
    return -np.sum(p*np.log(p))   
    
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

## fix it: the x axis may not be well aligned
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

def moments(p,x,k=2):
    M = np.zeros([k],dtype=float)
    for i in range(k):
        M[i] = p.dot(x**(i+1))
    return M

def moments_Y(Y,k=2):
    M = np.zeros([k],dtype=float)
    for i in range(k):
        M[i] = np.mean(Y**(i+1))
    return M

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

def dist_W1(p1,p2,x1,x2):
    mean1 = p1.dot(x1)
    mean2 = p2.dot(x2)
    x1 = x1/mean1
    x2 = x2/mean2
    return sp.stats.wasserstein_distance(x1, x2, u_weights=p1, v_weights=p2)

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
    

def pre_process(X,gene_name):
    ## filter 
    feature_select=np.ones(gene_name.shape,dtype=bool)
    temp=np.sum(X,axis=0)
    feature_select[temp<50]=0
    
    ## select the features
    X=np.log(X[:,feature_select]+1)
    gene_name=gene_name[feature_select]
    
    return X,gene_name
    
## some plotting functions
#def plot_density_1d(p,x):
#    M1,M2 = moments(p,x)
#    plt.plot(x,p,marker='o',label='mean:%s, var:%s'%(str(M1)[0:6],str(M2-M1**2)[0:6]))
#    #plt.bar(x,p,label='mean:%s, var:%s'%(str(M1)[0:6],str(M2-M1**2)[0:6]))
def plot_density_1d(p,x):
    M1,M2 = moments(p,x)
    X=np.random.choice(x, 2000, p=p)
    plt.figure(figsize=[18,5])
    plt.subplot(121)
    plt.plot(x,p,marker='.',lw=2,markersize=0,color="royalblue",label='mean:%s, var:%s'%(str(M1)[0:6],str(M2-M1**2)[0:6]))
    plt.fill_between(x, p, facecolor='royalblue', alpha=0.5)
    plt.plot(X, [-0.01]*len(X), '|', color='k',alpha=0.05)
    plt.xlim([-0.01,1])
    plt.subplot(122)
    plt.plot(x,p,marker='.',lw=2,markersize=0,color="royalblue",label='mean:%s, var:%s'%(str(M1)[0:6],str(M2-M1**2)[0:6]))
    plt.fill_between(x, p, facecolor='royalblue', alpha=0.5)
    plt.plot(X, [-0.01]*len(X), '|', color='k',alpha=0.05)
    plt.xlim([0.1,1])
    plt.ylim([0,1.2*np.max(p[x>0.1])])
    
    print('--- ccdf ---')
    for i in [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        print('>%s: %s%%'%(str(i),str(np.sum(p[x>i])*100)[0:5]))
    #plt.ylim([0,1.05*np.max(p)])    

def plot_density_2d(p,x,header='',range_=None):
    plt.scatter(x[:,0], x[:,1],s=5000*p,alpha=0.8,c=p,cmap='viridis')
    plt.colorbar()
    plt.title(header+'PC: %s,  MI: %s'%(str(pearson_corr(p,x))[0:6],str(mutual_info(p,x))[0:6]))
    if range_ is not None:
        plt.xlim(range_)
        plt.ylim(range_)
    
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
    
    
