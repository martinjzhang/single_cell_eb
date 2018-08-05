import numpy as np 
import matplotlib.pyplot as plt
from sc_deconv import *
from util import *
from sc_deconv_beta import *

def comparison_1d(p,x,tot_cts,n_degree=5,mean_cts=[10,100],known_gamma=False,zero_inflate=False):
    np.random.seed(42)
    mu_X = p.dot(x)
    B = int(tot_cts/mu_X)
    TD_list=[[int(mean_cts[0]/mu_X),int(tot_cts/mean_cts[0])],[int(mean_cts[1]/mu_X),int(tot_cts/mean_cts[1])]]
    log_err = np.zeros([len(TD_list),2],dtype=float)

    for i,(N_r,N_c) in enumerate(TD_list):
        print('### N_r=%s, mean_cts=%s, N_c=%s'%(str(N_r),str(int(N_r*mu_X)),str(N_c)))
        X,Y,data_info=data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=False)
        if known_gamma: 
            p_hat,dd_info=dd_1d(Y,noise='poi',c_reg=1e-6,n_degree=n_degree,zero_inflate=zero_inflate,verbose=False,gamma=N_r,debug_mode=False)
        else:
            p_hat,dd_info=dd_1d(Y,noise='poi',c_reg=1e-6,n_degree=n_degree,zero_inflate=zero_inflate,verbose=False,gamma=None,debug_mode=False)
        p_hat_ml,ml_info=ml_1d(Y)
        log_err[i,:]=np.log(plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info,verbose=True))
        print('\n')

    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(np.array(TD_list)[:,0],log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(np.array(TD_list)[:,0],log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('N_r')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.subplot(122)
    plt.plot(mean_cts,log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(mean_cts,log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('mean count')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.show()
    
def comparison_1d_beta(p,x,tot_cts,mean_cts=[10,100],verbose=False):
    np.random.seed(42)
    mu_X = p.dot(x)
    B = int(tot_cts/mu_X)
    TD_list=[[int(mean_cts[0]/mu_X),int(tot_cts/mean_cts[0])],[int(mean_cts[1]/mu_X),int(tot_cts/mean_cts[1])]]
    log_err = np.zeros([len(TD_list),2],dtype=float)

    for i,(N_r,N_c) in enumerate(TD_list):
        print('### N_r=%s, mean_cts=%s, N_c=%s'%(str(N_r),str(int(N_r*mu_X)),str(N_c)))
        X,Y,data_info=data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=True)
        p_hat,dd_info=dd_1d_beta(Y,N_r,K=5,verbose=verbose)
        p_hat_ml,ml_info=ml_1d(Y)
        log_err[i,:]=np.log(plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info,verbose=True))
        print('\n')

    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(np.array(TD_list)[:,0],log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(np.array(TD_list)[:,0],log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('N_r')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.subplot(122)
    plt.plot(mean_cts,log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(mean_cts,log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('mean count')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.show()

def tradeoff_1d(p,x,tot_cts,n_degree=5,mean_cts=[10,100],known_gamma=False,zero_inflate=False):
    np.random.seed(42)
    ### error plot with different trade-off
    mu_X = p.dot(x)
    B = int(tot_cts/mu_X)
    mu_Y = np.arange(1,100,5)
    TD_list= np.zeros([mu_Y.shape[0],2],dtype=int)
    TD_list[:,0] = (mu_Y/mu_X).clip(min=1)
    TD_list[:,1] = B/TD_list[:,0]
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(TD_list[:,0],TD_list[:,1])
    plt.xlabel('N_r')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.subplot(122)
    plt.plot(mu_Y, TD_list[:,1])
    plt.xlabel('mean count')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.show()

    # TD_list=[(25,int(B/25)),(100,int(B/100))]
    log_err = np.zeros([TD_list.shape[0],2],dtype=float)

    for i,(N_r,N_c) in enumerate(TD_list):
        #print('### N_r=%s, N_c=%s'%(str(N_r),str(N_c)))
        temp_err = np.zeros([rep_time,2])
        for j in range(rep_time):        
            X,Y,data_info=data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=False)
            if known_gamma: 
                p_hat,dd_info=dd_1d(Y,noise='poi',c_reg=1e-6,n_degree=n_degree,zero_inflate=zero_inflate,verbose=False,gamma=N_r,debug_mode=False)
            else:
                p_hat,dd_info=dd_1d(Y,noise='poi',c_reg=1e-6,n_degree=n_degree,zero_inflate=zero_inflate,verbose=False,gamma=None,debug_mode=False)                       
            p_hat_ml,ml_info=ml_1d(Y)
            temp_err[j,:]=np.log(plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info,verbose=False))
        log_err[i,:]=temp_err.mean(axis=0)
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(np.array(TD_list)[:,0],log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(np.array(TD_list)[:,0],log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('N_r')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.subplot(122)
    plt.plot(mu_Y,log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(mu_Y,log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('mean count')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.show()
    
def tradeoff_1d_beta(p,x,tot_cts,rep_time=10):
    np.random.seed(42)
    ### error plot with different trade-off
    mu_X = p.dot(x)
    B = int(tot_cts/mu_X)
    mu_Y = np.arange(1,100,5)
    TD_list= np.zeros([mu_Y.shape[0],2],dtype=int)
    TD_list[:,0] = (mu_Y/mu_X).clip(min=1)
    TD_list[:,1] = B/TD_list[:,0]
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(TD_list[:,0],TD_list[:,1])
    plt.xlabel('N_r')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.subplot(122)
    plt.plot(mu_Y, TD_list[:,1])
    plt.xlabel('mean count')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.show()

    # TD_list=[(25,int(B/25)),(100,int(B/100))]
    log_err = np.zeros([TD_list.shape[0],2],dtype=float)

    for i,(N_r,N_c) in enumerate(TD_list):
        #print('### N_r=%s, N_c=%s'%(str(N_r),str(N_c)))
        temp_err = np.zeros([rep_time,2])
        for j in range(rep_time):        
            X,Y,data_info=data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=False)
            p_hat,dd_info=p_hat,dd_info=dd_1d_beta(Y,N_r,K=6)                       
            p_hat_ml,ml_info=ml_1d(Y)
            temp_err[j,:]=np.log(plot_result_1d(p,p_hat,p_hat_ml,dd_info,ml_info,data_info,verbose=False))
        log_err[i,:]=temp_err.mean(axis=0)
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(np.array(TD_list)[:,0],log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(np.array(TD_list)[:,0],log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('N_r')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.subplot(122)
    plt.plot(mu_Y,log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(mu_Y,log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('mean count')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.show()
    
def tradeoff_moment_1d(p,x,tot_cts,k=2,test_moment=None,rep_time=10):
    np.random.seed(42)
    ### error plot with different trade-off
    mu_X = p.dot(x)
    B = int(tot_cts/mu_X)
    mu_Y = np.arange(1,100,5)
    TD_list= np.zeros([mu_Y.shape[0],2],dtype=int)
    TD_list[:,0] = (mu_Y/mu_X).clip(min=1)
    TD_list[:,1] = B/TD_list[:,0]
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(TD_list[:,0],TD_list[:,1])
    plt.xlabel('N_r')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.subplot(122)
    plt.plot(mu_Y, TD_list[:,1])
    plt.xlabel('mean count')
    plt.ylabel('N_c')
    plt.title('Buget B=%s, Total reads: %s'%(str(B),str(int(B*mu_X))))
    plt.show()

    log_err = np.zeros([TD_list.shape[0],2],dtype=float)
    M = moments(p,x,k=k)

    for i,(N_r,N_c) in enumerate(TD_list):
        temp_err = np.zeros([rep_time,2])
        for j in range(rep_time):        
            X,Y,data_info=data_gen_1d(p,x,N_c,N_r,noise='poi',verbose=False)           
            mean_hat_dd,var_hat_dd,M_hat_dd,N_r_hat = dd_moments_1d(Y,k=k,noise='poi')
            mean_hat_dd,var_hat_dd,M_hat_dd = M_convert(M_hat_dd,N_r_hat,N_r) 
            M_hat_ml = moments_Y(Y,k=k)
            _,_,M_hat_ml = M_convert(M_hat_ml,1,N_r) 
            
            if test_moment is None:
                temp_err[j,0] = np.log(np.mean(np.absolute(M_hat_dd-M)/np.absolute(M)))
                temp_err[j,1] = np.log(np.mean(np.absolute(M_hat_ml-M)/np.absolute(M)))
            else:
                temp_err[j,0] = np.log(np.mean(np.absolute(M_hat_dd[test_moment]-M[test_moment])/np.absolute(M[test_moment])))
                temp_err[j,1] = np.log(np.mean(np.absolute(M_hat_ml[test_moment]-M[test_moment])/np.absolute(M[test_moment])))
            
        log_err[i,:]=temp_err.mean(axis=0)
    if test_moment is not None: print('###### moments: ',np.array(test_moment)+1)
    plt.figure(figsize=[16,5])
    plt.subplot(121)
    plt.plot(np.array(TD_list)[:,0],log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(np.array(TD_list)[:,0],log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('N_r')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.subplot(122)
    plt.plot(mu_Y,log_err[:,0],marker='o',label='dd',color='darkorange')
    plt.plot(mu_Y,log_err[:,1],marker='o',label='ml',color='royalblue')
    plt.xlabel('mean count')
    plt.ylabel('log W1 error')
    plt.legend()
    plt.show()
    
def compare_moments_across_data(data_list,data_name_list,gene_list=None,size_factor=None,sub_samp=False,verbose=False):
    color_list = ['royalblue','navy','darkorange','darkred']

    n_gene = len(gene_list) if gene_list is not None else 15
    n_data = len(data_list)
    
    cv = np.zeros([n_gene,n_data],dtype=float)
    cv_ml = np.zeros([n_gene,n_data],dtype=float)
    relative_dif = np.zeros([n_gene,n_data],dtype=float)
    relative_dif_ml = np.zeros([n_gene,n_data],dtype=float)
    
    if gene_list is None: 
        random_gene = True
        for i in range(len(data_list)):
            if i==0:
                gene_list=set(data_list[i].var_names)
            else:
                gene_list=gene_list & set(data_list[i].var_names)
        gene_list=np.array(list(gene_list))
    else:
        random_gene = False
    
    i_gene_=0
    for i_gene,gene in enumerate(gene_list):
        if_continue=True
        if i_gene_==n_gene: break
            
        Y = {}       
        
        if sub_samp: n_sub = np.zeros([n_data],dtype=int)
            
        for i_data in range(n_data):
            data_ = data_list[i_data][:,gene]
            name_ = data_name_list[i_data]
            Y[name_] = np.array(data_.X,dtype=int)
            if random_gene is True and (Y[name_].mean()<0.1): 
                if_continue=False
                continue
            if sub_samp: n_sub[i_data] = np.array(data_.X,dtype=int).sum()
                
        if sub_samp: n_sub = n_sub.min()
            
        if if_continue:
            if verbose: print('## %s: '%gene)
        else:
            continue
                     
        Y = {} 
        for i_data in range(n_data):
            data_ = data_list[i_data][:,gene]
            name_ = data_name_list[i_data]
            
            Y[name_] = np.array(data_.X,dtype=int)
            if sub_samp: Y[name_] = sub_sample(Y[name_],n_sub)         
            
            sf = size_factor[i_data].clip(min=0.1) if size_factor is not None else None
            
            
            m_,var_,M_,_=dd_moments_1d(Y[name_],gamma=1,size_factor=sf)
            cv[i_gene_,i_data] = np.sqrt(var_)/m_
            cv_ml[i_gene_,i_data] = np.std(Y[name_])/np.mean(Y[name_]) if sf is None else np.std(Y[name_]/sf)/np.mean(Y[name_]/sf)
            relative_dif[i_gene_,i_data] = (cv[i_gene_,i_data]-cv[i_gene_,0])/cv[i_gene_,0]
            relative_dif_ml[i_gene_,i_data] = (cv_ml[i_gene_,i_data]-cv_ml[i_gene_,0])/cv_ml[i_gene_,0]
            if verbose:
                print('# %s'%name_)
                print('mean=%0.4f, var=%0.4f, cv=%0.4f, cv_ml=%0.4f, B=%d'\
                      %(m_,var_,cv[i_gene_,i_data],cv_ml[i_gene_,i_data],int(m_*Y[name_].shape[0])))
        if verbose: print('\n')        
        
        i_gene_ +=1
                
    plt.figure(figsize=[18,5])
    for i_data in range(n_data):
        name_ = data_name_list[i_data]
        plt.bar(np.arange(n_gene)+0.1+0.3*i_data,cv[:,i_data],width=0.15,color=color_list[i_data*2],label=name_)
        plt.bar(np.arange(n_gene)+0.25+0.3*i_data,cv_ml[:,i_data],width=0.15,color=color_list[i_data*2+1],label='ml_'+name_)
    plt.xticks(np.arange(n_gene)+0.25,gene_list,rotation=45)
    plt.title('estimated cv (std/mean)')
    plt.legend()
    plt.show()

    plt.figure(figsize=[18,5])
    for i_data in range(1,n_data):
        name_ = data_name_list[i_data]
        plt.bar(np.arange(n_gene)+0.1+0.3*i_data,relative_dif[:,i_data],width=0.3,color=color_list[i_data*2],label=name_)
        plt.bar(np.arange(n_gene)+0.4+0.3*i_data,relative_dif_ml[:,i_data],width=0.3,color=color_list[i_data*2+1],label='ml_'+name_)
    plt.xticks(np.arange(n_gene)+0.25,gene_list,rotation=45)
    plt.title('relative difference')
    plt.legend()
    plt.show()
    
    print('## summary')
    print('# average cv_68k:%0.4f'%cv[:,1].mean())
    print('# average cv_68k/cv_ml_68k:%0.4f'%(cv[:,1]/cv_ml[:,1]).mean())
    print('# average cv_68k/cv_8k:%0.4f'%(cv[:,1]/cv[:,0]).mean())
    print('# average cv_ml_68k/cv_ml_8k:%0.4f'%(cv_ml[:,1]/cv_ml[:,0]).mean())
    print('# average relative difference: %0.4f'%np.absolute(relative_dif).mean())    
    print('# average relative ml difference: %0.4f'%np.absolute(relative_dif_ml).mean())