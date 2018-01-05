import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvx
import time
from util import *
from module_deconv import *

## testing bench 
def simu_dd(p,x,param_list,rep_time,output_folder):
    res_dd=np.zeros([param_list.shape[0],rep_time],dtype=float)
    res_ml=np.zeros([param_list.shape[0],rep_time],dtype=float)
    B_list=param_list[:,2]
    
    for i in range(param_list.shape[0]):
        N_c=param_list[i,0]
        N_r=param_list[i,1]
        for j in range(rep_time):
            Y_pdf,Y_supp=data_gen_1d(p,x,N_c,N_r,noise='poi')
            p_hat_dd,dd_info=deconv_1d(Y_pdf,Y_supp,x,N_c,N_r,noise='poi',opt='dd')
            p_hat_ml=esti_ml(Y_pdf,Y_supp,x,N_c,N_r)
            if p_hat_dd is not None:
                res_dd[i,j]=dd_evaluation(p,p_hat_dd)
            else:
                print 'error'
            res_ml[i,j]=dd_evaluation(p,p_hat_ml)
                
                
    ## store the results
    plt.figure()
    plt.plot(x,p,marker='o')
    plt.savefig(output_folder+'/true_dist.png')
    plt.close()
    
    np.save(output_folder+'/res.npy',(res_dd,res_ml,p,x,param_list))
    
    err_dd=np.mean(res_dd,axis=1)
    err_ml=np.mean(res_ml,axis=1)
    vmin=np.min([np.min(err_dd),np.min(err_ml)])
    vmax=np.max([np.max(err_dd),np.max(err_ml)])
    
    best_param=[]
    for B in np.unique(B_list):
        temp=np.argmin(err_dd[B_list==B])
        best_param.append(list(param_list[B_list==B,:][temp,0:2]))
    best_param=np.array(best_param)
    
    
    plt.figure(figsize=[20,8])    
    plt.subplot(1,2,1)
    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=1000, c=err_dd,cmap='viridis',vmin=0, vmax=0.4)
    plt.plot(np.log(best_param[:,0]),best_param[:,1],marker='o',color='r')
    plt.xlabel('log Nc')
    plt.ylabel('Nr')
    plt.title('density deconvolution')
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=1000, c=err_ml,cmap='viridis',vmin=0, vmax=1)
    plt.xlabel('log Nc')
    plt.ylabel('Nr')
    plt.title('maximum likelihood')
    plt.colorbar()
    plt.savefig(output_folder+'/error_plot.png')
    plt.close()   
    return res_dd,res_ml

## testing bench 
def simu_dd_2d(p,x,param_list,rep_time,output_folder):
    res_dd=np.zeros([param_list.shape[0],rep_time],dtype=float)
    res_ml=np.zeros([param_list.shape[0],rep_time],dtype=float)
    B_list=param_list[:,2]
    
    for i in range(param_list.shape[0]):
        for j in range(rep_time):
            N_c=param_list[i,0]
            N_r=param_list[i,1]
            while True:                
                try: 
                    res = exp_2d(p,x,(N_c,N_r,'mml',0.1,0))
                    res_dd[i,j]=res['err_tv_dd']
                    res_ml[i,j]=res['err_tv_ml']
                    break
                except:              
                    print 'exception, do it again!'            
                
    ## store the results
    plt.figure()
    plot_density_2d(p,x)
    plt.savefig(output_folder+'/true_dist.png')
    plt.close()
    
    np.save(output_folder+'/res.npy',(res_dd,res_ml,p,x,param_list))
    
    err_dd=np.mean(res_dd,axis=1)
    err_ml=np.mean(res_ml,axis=1)
    vmin=np.min([np.min(err_dd),np.min(err_ml)])
    vmax=np.max([np.max(err_dd),np.max(err_ml)])
    
    best_param=[]
    for B in np.unique(B_list):
        temp=np.argmin(err_dd[B_list==B])
        best_param.append(list(param_list[B_list==B,:][temp,0:2]))
    best_param=np.array(best_param)
    
    
    plt.figure(figsize=[20,8])    
    plt.subplot(1,2,1)
    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=20000*err_dd**2, c=err_dd,cmap='viridis',vmin=0, vmax=0.4,alpha=0.8)
    plt.plot(np.log(best_param[:,0]),best_param[:,1],marker='o',color='r')
    plt.xlabel('log Nc')
    plt.ylabel('Nr')
    plt.title('density deconvolution')
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.scatter(np.log(param_list[:,0]),param_list[:,1], s=20000*err_ml**2, c=err_ml,cmap='viridis',vmin=0, vmax=1,alpha=0.8)
    plt.xlabel('log Nc')
    plt.ylabel('Nr')
    plt.title('maximum likelihood')
    plt.colorbar()
    plt.savefig(output_folder+'/error_plot.png')
    plt.close()   
    return res_dd,res_ml

def exp_2d(p,x,param): 
    # param: the parameter tuple with (N_c,N_r,option,lamb,vis)
    N_c    = param[0]
    N_r    = param[1]
    option = param[2]
    lamb   = param[3]
    vis    = param[4]
    print 'N_c:%s, N_r:%s, option:%s, lamb:%s'%(str(N_c),str(N_r),option,str(lamb))
    
    #generating the data 
    Y_pdf,Y_supp = data_gen_poi_2d(p,x,N_c,N_r)
    if vis == 1:
        plt.figure(figsize=[12,5])
        plt.subplot(1,2,1)
        plot_density_2d(p,x)
        plt.subplot(1,2,2)
        plot_density_2d(Y_pdf,Y_supp)
        plt.show()
                              
    #recover the probability
    p_hat,dd_info = dd_2d(Y_pdf,Y_supp,x,N_c,N_r,lamb=lamb,option=option)
    p_hat_ml      = esti_ml(Y_pdf,Y_supp,x,N_c,N_r) 
    
    if vis == 1:
        _=plot_dd_result(p,p_hat,dd_info)
    
    res              = {}    
    res['p']         = p
    res['x']         = x
    res['p_hat']     = p_hat
    res['p_hat_ml']  = p_hat_ml
    res['true_pc']   = pearson_corr(p,x)
    res['true_mi']   = mutual_info(p,x)
    res['dd_pc']     = pearson_corr(p_hat,x)
    res['dd_mi']     = mutual_info(p_hat,x)
    res['ml_pc']     = pearson_corr(p_hat_ml,x)
    res['ml_mi']     = mutual_info(p_hat_ml,x)    
    res['err_tv_dd'] = dist_tv(p,p_hat)
    res['err_tv_ml'] = dist_tv(p,p_hat_ml)
    res['err_pc_dd'] = np.absolute((res['dd_pc']-res['true_pc'])/res['true_pc']*100)
    res['err_pc_ml'] = np.absolute((res['ml_pc']-res['true_pc'])/res['true_pc']*100)
    res['err_mi_dd'] = np.absolute((res['dd_mi']-res['true_mi'])/res['true_mi']*100)
    res['err_mi_ml'] = np.absolute((res['ml_mi']-res['true_mi'])/res['true_mi']*100)
    
    return res