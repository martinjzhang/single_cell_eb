import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import itertools


class BsplineND():
    def __init__(self, knots, degree=3, periodic=False):
        """
        :param knots: a list of the spline knots with ndim = len(knots)
        """
        self.ndim = len(knots)
        self.splines = []
        self.knots = knots
        self.degree = degree
        for idim, knots1d in enumerate(knots):
            nknots1d = len(knots1d)
            y_dummy = np.zeros(nknots1d)
            knots1d, coeffs, degree = si.splrep(knots1d, y_dummy, k=degree,
                                                per=periodic)
            self.splines.append((knots1d, coeffs, degree))
        self.ncoeffs = [len(coeffs) for knots, coeffs, degree in self.splines]

    def evaluate(self, position):
        """
        :param position: a numpy array with size [ndim, npoints]
        :returns: a numpy array with size [nspl1, nspl2, ..., nsplN, npts]
                  with the spline basis evaluated at the input points
        """
        ndim, npts = position.shape

        values_shape = self.ncoeffs + [npts]
        values = np.empty(values_shape)
        ranges = [range(icoeffs) for icoeffs in self.ncoeffs]
        for icoeffs in itertools.product(*ranges):
            values_dim = np.empty((ndim, npts))
            for idim, icoeff in enumerate(icoeffs):
                coeffs = [1.0 if ispl == icoeff else 0.0 for ispl in
                          range(self.ncoeffs[idim])]
                values_dim[idim] = si.splev(
                        position[idim],
                        (self.splines[idim][0], coeffs, self.degree))

            values[icoeffs] = np.product(values_dim, axis=0)
        return values

def Q_gen_ND(points=None,n_degree=5,opt='1d',zero_inflate=False,verbose=False):        
    if opt=='1d':
        if points is None: points = np.linspace(0,1,101)
        knotsx = np.arange(n_degree)/(n_degree-1)
        bspline1d = BsplineND([knotsx], periodic=False)
        values1d = bspline1d.evaluate(points[None, :])
        Q = (values1d[np.linalg.norm(values1d,axis=1)>1e-6,:]).T
        
        if zero_inflate:
            Q_t        = np.zeros([Q.shape[0],Q.shape[1]+1],dtype=float)
            Q_t[0,0]   = 1
            #Q_t[:,0]  = np.exp(-100*points)
            Q_t[:,1:] = Q
            Q = Q_t
            n_degree += 1
        
        if verbose:
            plt.figure(figsize=[16,5])
            for i in range(Q.shape[1]):
                plt.plot(points,Q[:,i],label=str(i+1))
            plt.legend()
            plt.show()        
        return Q,n_degree
    
    elif opt=='2d':
        if points is None: 
            tempx,tempy = np.meshgrid(np.linspace(0,1,101), np.linspace(0,1,101), indexing='ij')
            npt = tempx.shape[0]
            points = np.array([tempx.flatten(), tempy.flatten()])
        knots = [np.arange(n_degree)/(n_degree-1), np.arange(n_degree)/(n_degree-1)]
        
        bspline2d = BsplineND(knots, periodic=False)
        values2d = bspline2d.evaluate(points)
        
        values2d = np.reshape(values2d,[values2d.shape[0]*values2d.shape[1],values2d.shape[2]])      
        Q = (values2d[np.linalg.norm(values2d,axis=1)>1e-6,:]).T
        
        if verbose:
            plt.figure(figsize=[16,16])           
            for icol in range(n_degree):
                for irow in range(n_degree):
                    temp = icol*n_degree+irow+1
                    plt.subplot(n_degree,n_degree,temp)                    
                    plt.imshow(Q[:,temp-1].reshape(npt, npt).T)
            plt.suptitle('2D Bspline basis from scipy (non-periodic)')
            plt.show()
        return Q,n_degree    