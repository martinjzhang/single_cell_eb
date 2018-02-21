import numpy as np 
import matplotlib.pyplot as plt
from sc_deconv import *
from util import *

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

def tradeoff_1d(p,x,tot_cts,rep_time=10,n_degree=7,known_gamma=False,zero_inflate=False):
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