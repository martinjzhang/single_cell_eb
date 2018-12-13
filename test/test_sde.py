""" Test for the core code of the method module
"""
import numpy as np
import sceb.scdd_extra as sde

def load_test_toy_data(Nc=10000, Nr=1):
    np.random.seed(0)
    x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    p = np.array([0.4, 0.1, 0.1, 0.4], dtype=float)
    temp = np.random.choice(np.arange(x.shape[0]),
                            Nc, p=p, replace=True)  
    X = x[temp]
    size_factor = (np.random.randn(Nc)*0.2 + 1).clip(min=0.5)
    Y = np.random.poisson((X.T*size_factor).T*Nr)
    return Y,X,size_factor

def load_test_toy_data_fish(Nc=10000, Nr=1):
    np.random.seed(0)
    x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    p = np.array([0.4, 0.1, 0.1, 0.4], dtype=float)
    temp = np.random.choice(np.arange(x.shape[0]),
                            Nc, p=p, replace=True)  
    X = x[temp]
    size_factor = (np.random.randn(Nc)*0.2 + 1).clip(min=0.5)
    Y = (X.T*size_factor).T*Nr
    return Y,X,size_factor 

def test_M_single():
    """ Test sde.M_single()
    """
    Nr = 3
    Y,X,size_factor = load_test_toy_data(Nc=10000, Nr=Nr)
    M1_gt = np.mean(X, axis=0)[0]
    M2_gt = np.mean(X**2, axis=0)[0]
    M_ml, M_dd = sde.M_single(Y[:, 0], size_factor=size_factor, Nr=Nr)
    print('M1_gt=%0.4f, M2_gt=%0.4f, M1_dd=%0.4f, M2_dd=%0.4f'%(
           M1_gt, M2_gt, M_dd[0], M_dd[1]))
    assert np.absolute(M1_gt-M_dd[0]) < 0.01
    assert np.absolute(M2_gt-M_dd[1]) < 0.01
    
def test_cv_single():
    """ Test sde.cv_single()
    """
    Nr = 3
    Y,X,size_factor = load_test_toy_data(Nc=10000, Nr=Nr)
    cv_gt = (np.std(X, axis=0)/np.mean(X, axis=0))[0]
    cv_ml, cv_dd = sde.cv_single(Y[:,0], size_factor=size_factor)
    print('cv_gt=%0.4f, cv_ml=%0.4f, cv_dd=%0.4f'%(
           cv_gt, cv_ml, cv_dd))
    assert np.absolute(cv_gt-cv_dd) < 0.01
    
def test_cv_single_fish():
    """ Test sde.cv_single_fish()
    """
    Nr = 3
    Y,X,size_factor = load_test_toy_data_fish(Nc=10000, Nr=Nr)
    cv_gt = (np.std(X, axis=0)/np.mean(X, axis=0))[0]
    cv_ml, cv_dd = sde.cv_single(Y[:,0], size_factor=size_factor)
    cv_fish = sde.cv_single_fish(Y[:,0], size_factor=size_factor)
    print('cv_gt=%0.4f, cv_fish=%0.4f'%(
           cv_gt, cv_fish))
    assert np.absolute(cv_gt-cv_fish) < 0.01
    
def test_PC_single():
    """ Test sde.PC_single()
    """
    Nr = 5
    Y,X,size_factor = load_test_toy_data(Nc=10000, Nr=Nr)
    PC_gt = np.corrcoef(X.T)[0,1]
    PC_ml, PC_dd = sde.PC_single(Y[:,0], Y[:,1], size_factor=size_factor)
    print('PC_gt=%0.4f, PC_ml=%0.4f, PC_dd=%0.4f'%(
           PC_gt, PC_ml, PC_dd))
    assert np.absolute(PC_gt-PC_dd) < 0.01

def test_PC_single_fish():
    """ Test sde.PC_single_fish()
    """
    Nr = 5
    Y,X,size_factor = load_test_toy_data_fish(Nc=10000, Nr=Nr)
    PC_gt = np.corrcoef(X.T)[0,1]
    PC_ml, PC_dd = sde.PC_single(Y[:,0], Y[:,1], size_factor=size_factor)
    PC_fish = sde.PC_single_fish(Y[:,0], Y[:,1], size_factor=size_factor)
    print('PC_gt=%0.4f, PC_fish=%0.4f'%(
           PC_gt, PC_fish))
    assert np.absolute(PC_gt-PC_fish) < 0.01