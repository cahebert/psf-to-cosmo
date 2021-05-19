"""
Module containing the AstrometryField class for analyzing astrometric residual fields.
"""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn
seaborn.set_palette('bright')

import treecorr

#import GPy
#import gpflow
#import tensorflow as tf

#-------------------------------------------------------------------------------

#class AnisotropicRBF(gpflow.kernels.AnisotropicStationary):
#    def K_d(self, d):
#        #def __init__(self, variance=1.0, lengthscales=1.0, rho=1.0, active_dims=None):
#        #    super().__init__(variance=variance, lengthscales=lengthscales, active_dims=active_dims)
#        #    self.rho = Parameter(rho, transform=positive())
#        return self.variance * tf.exp(-0.5*tf.tensordot(d, d, 2))

#-------------------------------------------------------------------------------

class AstrometryField:
    def __init__(self, infile, pscale=0.2, bins=10):
        """
        Initialize AstrometryField class.

        Parameters
        ----------
         ddict: dict,
             Dictionary output from psfws
         pscale: float, 0.2,
             Pixel scale in asec (0.2 for LSST)
         bins: int, optional
             The number of distance bins at which to compute the 2-point
             correlation functions

        Returns
        -------
         None
        """
        self.ddict = self.load_data(infile)

        self.x = np.array(self.ddict['x'])
        self.y = np.array(self.ddict['y'])
        self.thx = np.array(self.ddict['thx'])
        self.thy = np.array(self.ddict['thy'])

        self.pscale = pscale
        self.res_x, self.res_y = self.get_astrometric_residuals()

        self.xs = np.array([self.thx, self.thy]).T
        self.ys = np.array([self.res_x, self.res_y]).T

        self.bins = bins
        self.dr, self.xi0, self.xi1 = self.compute_2pcf(self.xs, self.ys, bins=self.bins)

    def load_data(self, infile):
        """
        Load astrometric field data from file

        Parameters
        ----------
         infile: str,
             Input file to analyze

        Returns
        -------
         ddict: dict,
             Dictionary of astrometric field data
        """
        ddict = pickle.load(open(infile, 'rb'))
        return ddict

    def get_astrometric_residuals(self):
        """
        Compute the x- and y-components of the astrometric residuals by
        subtracting the mean of each field, respectively.

        Parameters
        ----------
         None

        Returns
        -------
         res_x: ndarray,
             x-component of the astrometric residual field
         res_y: ndarray,
             y-component of the astrometric residual field
        """
        # The measurements are done per psf by simulating a 50x50 pixel image
        # on the sky centered at some location thx, thy. 
        # 
        # The centroid (x,y) is measured in pixels relative to bottom corner of
        # this image. To get the residuals, subtract 25 pixels from the x,y
        # values to shift the origin of this 50x50 image to thx, thy; that's it!
        # Now x and y are the residuals. Convert from pixels to angular distance
        # using the pixel scale: here using the LSST pixel scale 0.2 to get
        # units of arcseconds.

        # astrometric residuals:
        res_x = [(x-25) * self.pscale for x in self.x]
        res_y = [(y-25) * self.pscale for y in self.y]

        # subtract 1st order polynomial, ie mean:
        # according the PF this should be a third order polynomial! to-do.
        res_x = np.asarray([x-np.mean(res_x) for x in res_x])
        res_y = np.asarray([y-np.mean(res_y) for y in res_y])

        return res_x, res_y

    def get_train_test(self, size):
        """
        Split the astrometric residual field into training and testing subsets.

        Parameters
        ----------
         size: int,
             Size (number) of points in the training subset.

        Returns
        -------
         xs_train: ndarray,
             Training subset of the field components
         xs_test: ndarray,
             Testing subset of the field components
         ys_train: ndarray,
             Training subset of the astrometric residual components
         ys_test: ndarray,
             Testing subset of the astrometric residual components
        """
        xs_train, xs_test, ys_train, ys_test = train_test_split(self.xs, self.ys,
                                                                test_size=size,
                                                                random_state=42)

        return xs_train, xs_test, ys_train, ys_test

    def compute_2pcf(self, xs, ys, bins=10):
        """
        Compute the 2-point correlation function of each component of the
        astrometric residual field.

        Parameters
        ----------
         xs: ndarray,
             List of the x- and y-components of the field
         ys: ndarray,
             List of the x- and y-components of the astrometric residual field
         bins: int, optional
             The number of distance bins at which to compute the 2-point
             correlation functions

        Returns
        -------
         dr: ndarray,
             separations at which the 2-point correlation functions were calculated
         xi0: ndarray,
             2-point correlation function of the x-component of the astrometric
             residual field
         xi1: ndarray,
             2-point correlation function of the y-component of the astrometric
             residual field
        """
        # ``upper triangle'' indices for an (N, N) array
        ind = np.triu_indices(xs.shape[0])

        # seps has shape (N, N) up to ind
        # calculate the euclidean separation between each point in the field
        seps = np.asarray([np.sqrt(np.square(x[0]-xs[:,0])+np.square(x[1]-xs[:,1])) for x in xs])[ind]

        # pps0, pps1 have shape (N, N) up to ind
        # calculate the pair products of each component of each point of the
        # astrometric residuals
        pps0 = np.outer(ys[:,0], ys[:,0])[ind]
        pps1 = np.outer(ys[:,1], ys[:,1])[ind]

        # Use histograms to efficiently select pps according to sep
        # Inspired by Gary Bernstein via Pierre-Francois Leget
        counts, dr = np.histogram(seps, bins=bins)
        xi0, _ = np.histogram(seps, bins=bins, weights=pps0)
        xi1, _ = np.histogram(seps, bins=bins, weights=pps1)

        dr = 0.5*(dr[:-1]+dr[1:])
        xi0 /= counts
        xi1 /= counts

        return dr, xi0, xi1

    def plot_2pcf(self, ax, dr, xi0, xi1):
        """
        Plot the two-point correlation function for the x- and y-components of
        the astrometric residual field as a function of distance between points.

        Parameters
        ----------
         ax: axis,
             Matplotlib axis in which to plot
         dr: ndarray,
             separations at which the 2-point correlation functions were calculated
         xi0: ndarray,
             2-point correlation function of the x-component of the astrometric
             residual field
         xi1: ndarray,
             2-point correlation function of the y-component of the astrometric
             residual field

        Returns
        -------
         None
        """
        # Plot the two-point correlation functions as a function of distance
        ax.axhline(y=0, ls='--', lw=1, c='gray')
        ax.plot(dr, xi0, marker='o', ms=5, ls='-', lw=1, label=r'$\xi_{xx}$')
        ax.plot(dr, xi1, marker='o', ms=5, ls='-', lw=1, label=r'$\xi_{yy}$')
        ax.legend()
        ax.set_xlabel(r'$\Delta$ [degrees]', fontsize=12);
        ax.set_ylabel(r'$\xi(\Delta)$ [degrees$^2$]', fontsize=12)

    def plot_astrometric_residuals(self, ax, xs, ys):
        """
        Plot the astrometric residual field of a set of points.

        Parameters
        ----------
         ax: axis,
             Matplotlib axis in which to plot
         xs: ndarray,
             List of the x- and y-components of the field
         ys: ndarray,
             List of the x- and y-components of the astrometric residual field

        Returns
        -------
         None
        """
        qdict = dict(
            alpha=1,
            angles='uv',
            headlength=5,
            headwidth=3,
            headaxislength=4,
            minlength=0,
            pivot='middle',
            scale_units='xy',
            width=0.002,
            color='#001146'
        )
        
        q = ax.quiver(xs[:,0], xs[:,1], ys[:,0], ys[:,1], scale=1, **qdict)
        ax.quiverkey(q, 0.0, 1.8, 0.1, 'residual = 0.1 arcsec',
                     coordinates='data', labelpos='N',
                     color='darkred', labelcolor='darkred')
        
        ax.set_xlabel('[degrees]', fontsize=12);
        ax.set_ylabel('[degrees]', fontsize=12)

        ax.set_xlim(-1.95, 1.95)
        ax.set_ylim(-1.9, 2.0)
        ax.set_aspect('equal')

    def compute_2pcf_tc(self, xs, ys, bins=10):
        """
        Compute the 2-point correlation (covariance) function using TreeCorr.

        Parameters
        ----------
         xs: ndarray,
             List of the x- and y-components of the field
         ys: ndarray,
             List of the x- and y-components of the astrometric residual field
         bins: int, optional
             The number of distance bins at which to compute the 2-point
             correlation functions

        Returns
        -------
         ???: ???,
             ???
        """

        seps = self.compute_separations(xs)

        cat0 = treecorr.Catalog(x=xs[:,0], y=xs[:,1], k=ys[:,0])
        cat1 = treecorr.Catalog(x=xs[:,0], y=xs[:,1], k=ys[:,1])

        kk0 = treecorr.KKCorrelation(min_sep=1e-10, max_sep=np.max(seps), nbins=bins, bin_type='Linear')
        kk1 = treecorr.KKCorrelation(min_sep=1e-10, max_sep=np.max(seps), nbins=bins, bin_type='Linear')

        kk0.process(cat0)
        kk1.process(cat1)

        return kk0, kk1


    #def gpr_train(self, xs, ys):
    #    #k = gpflow.kernels.SquaredExponential(lengthscales=[0.01, 0.01]) + gpflow.kernels.White()
    #    k = gpflow.kernels.SquaredExponential(active_dims=[0]) \
    #      * gpflow.kernels.SquaredExponential(active_dims=[1]) \
    #      + gpflow.kernels.White()
    #    #k = AnisotropicRBF() + gpflow.kernels.White()
    #    
    #    m = gpflow.models.GPR(data=(np.array(xs).T,
    #                                np.array(ys).T),
    #                          kernel=k,
    #                          mean_function=None)
    #    
    #    opt = gpflow.optimizers.Scipy()
    #    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))
    #    
    #    gpflow.utilities.print_summary(m)

    #    return m

    #def gpr_predict(self, m, xs, ys):
    #    pred, var = m.predict_f(np.array([xs[0], xs[1]]).T)

    #    return pred, var


#-------------------------------------------------------------------------------

if __name__ == '__main__':
    """
    Main function for testing and debugging purposes.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile',type=str,required=False,default='output/out_1.pkl',
                        help='Input data file [.pkl]')
    parser.add_argument('--size',type=int,required=False,default=10,
                        help='Size of training sample [int]')
    args = parser.parse_args()

    #---------------------------------------------------------------------------
    
    af = AstrometryField(args.infile)

    #---------------------------------------------------------------------------

    # Sanity check plot: 2-point correlation function
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    af.plot_astrometric_residuals(axs[0], af.xs, af.ys)
    af.plot_2pcf(axs[1], af.dr, af.xi0, af.xi1)

    plt.suptitle(f'{os.path.basename(args.infile)}')
    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------------------------

    ##m = af.gpr_train([xs0_train, xs1_train], [ys0_train, ys1_train])
    ###Xs0, Xs1 = np.meshgrid(xs0_train, xs1_train)
    ###Xs0, Xs1 = np.meshgrid(xs0_test, xs1_test)
    ##Xs0, Xs1 = np.meshgrid(np.linspace(np.min(af.thx), np.max(af.thx), 100),
    ##                       np.linspace(np.min(af.thy), np.max(af.thy), 100))
    ##pred, var = af.gpr_predict(m, [Xs0, Xs1], [ys0_test, ys1_test])
    ##
    ##fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    ##af.plot_astrometric_residuals(axs[0], [af.thx, af.thy], [af.res_x, af.res_y])
    ##af.plot_astrometric_residuals(axs[1], [Xs0, Xs1], [pred[:,:,0], pred[:,:,1]])
    ##
    ##plt.show()
