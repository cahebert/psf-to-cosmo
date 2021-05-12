"""
Module containing the AstrometryField class for analyzing astrometric residual fields.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, ddict):
        """
        Initialize AstrometryField class.

        Parameters
        ----------
         ddict: dict,
             Dictionary output from psfws

        Returns
        -------
         None
        """
        self.ddict = ddict

        self.x = np.array(out['x'])
        self.y = np.array(out['y'])
        self.thx = np.array(out['thx'])
        self.thy = np.array(out['thy'])

        self.pscale = 0.2 # LSST pixel scale in asec
        self.res_x, self.res_y = self.get_astrometric_residuals()

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
        res_x = np.array([(x-25) * self.pscale for x in self.x])
        res_y = np.array([(y-25) * self.pscale for y in self.y])

        # subtract 1st order polynomial, ie mean:
        # according the PF this should be a third order polynomial! to-do.
        res_x = np.array([x-np.mean(res_x) for x in res_x])
        res_y = np.array([y-np.mean(res_y) for y in res_y])

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
         thx_train: ndarray,
             Training subset of the x-component of the field
         thy_train: ndarray,
             training subset of the y-component of the field
         res_x_train: ndarray,
             Training subset of the x-component of the astrometric
             residual field
         res_y_train: ndarray,
             Training subset of the y-component of the astrometric
             residual field
         thx_test: ndarray,
             Testing subset of the x-component of the field
         thy_test: ndarray,
             Testing subset of the y-component of the field
         res_x_test: ndarray,
             Testing subset of the x-component of the astrometric
             residual field
         res_y_test: ndarray,
             Testing subset of the y-component of the astrometric
             residual field
        """
        # Randomly select `size` unique indices for the x- and y-components
        rng = np.random.default_rng()
        x_i = rng.choice(list(range(len(self.thx))), size=size, replace=False)
        y_i = rng.choice(list(range(len(self.thy))), size=size, replace=False)

        return self.thx[x_i], self.thy[y_i], self.res_x[x_i], self.res_y[y_i], \
               self.thx[~x_i], self.thy[~y_i], self.res_x[~x_i], self.res_y[~y_i]

    def compute_separations(self, xs):
        """
        Compute (euclidean) separations between each point in the field.

        Parameters
        ----------
         xs: list,
             List of the x- and y-components of the field (each an array)

        Returns
        -------
         seps: ndarray,
             Array of distances between each point
        """
        # SM: pts should probably be defined outside of the function...
        pts = np.array([xs[0], xs[1]]).T

        # Compute the (euclidean) separation between each point and the field of points
        seps = np.array([np.sqrt(np.square(pt[0]-pts[:,0])+np.square(pt[1]-pts[:,1])) for pt in pts])

        return seps

    def select_values(self, xs, ys, bins):
        """
        Select the values of the points whose separation lie within the bins.

        Parameters
        ----------
         xs: list,
             List of the x- and y-components of the field (each an array)
         ys: list,
             List of the x- and y-components of the astrometric residual field
             (each an array)
         bins: list,
             List of endpoints of the distance bin

        Returns
        -------
         vals: list,
             List of vals whose separation lie within the bins; return nans if
             there are no such vals
        """
        seps = self.compute_separations(xs)
        vals = np.array([ys[0], ys[1]]).T

        return np.vstack([vals[((bins[0] <= sep) & (sep < bins[1]))]
                          if np.sum((bins[0] <= sep) & (sep < bins[1])) > 0
                          else np.array([np.nan, np.nan])
                          for sep in seps])

    def compute_2pcf(self, vals):
        """
        Compute the 2-point correlation (covariance) function of a set of values.

        Parameters
        ----------
         vals: list,
             List of vals

        Returns
        -------
         pcfs: ndarray,
             2-point correlation functions for the x- and y-components of the
             input values
        """
        # Compute pair products for every point (2D)
        # Method adapted from https://stackoverflow.com/questions/62012339/efficiently-computing-all-pairwise-products-of-a-given-vectors-elements-in-nump
        # SM: is there a more elegant/efficient way to compute these pair products?
        pcfs0 = np.nanmean(np.outer(vals[:,0], vals[:,0])[~np.tri(len(vals),k=-1,dtype=bool)])
        pcfs1 = np.nanmean(np.outer(vals[:,1], vals[:,1])[~np.tri(len(vals),k=-1,dtype=bool)])
        pcfs = np.array([pcfs0, pcfs1])
        
        return pcfs

    def plot_2pcf(self, ax, xs, ys, bins=10):
        """
        Plot the two-point correlation function for the x- and y-components of
        the astrometric residual field as a function of distance between points.

        Parameters
        ----------
         ax: axis,
             Matplotlib axis in which to plot
         xs: list,
             List of the x- and y-components of the field (each an array)
         ys: list,
             List of the x- and y-components of the astrometric residual field
             (each an array)
         bins: int, optional
             The number of distance bins at which to compute the 2-point
             correlation functions.

        Returns
        -------
         None
        """
        # Compute the separations between each point in the field
        seps = self.compute_separations(xs)

        # Define distance bins across the range of separation values
        rs = np.histogram_bin_edges(seps, bins=bins)
        bins = np.array([rs[0:-1], rs[1:]]).T

        # For each bin, select the astrometric-residual values of the points
        # that lie within that distance bin
        vals = [self.select_values(xs, ys, b) for b in bins]

        # Compute the 2-point correlation function for each bin
        pcfs = np.array([self.compute_2pcf(val) for val in vals])

        # Plot the two-point correlation functions as a function of distance
        ax.scatter(0.5*(bins[:,0]+bins[:,1]), pcfs[:,0], label=r'$\xi_{xx}$')
        ax.scatter(0.5*(bins[:,0]+bins[:,1]), pcfs[:,1], label=r'$\xi_{yy}$')
        ax.legend()
        ax.set_xlabel(r'$\Delta$', fontsize=12);
        ax.set_ylabel(r'$\xi(\Delta)$', fontsize=12)

    def plot_astrometric_residuals(self, ax, xs, ys):
        """
        Plot the astrometric residual field of a set of points.

        Parameters
        ----------
         ax: axis,
             Matplotlib axis in which to plot
         xs: list,
             List of the x- and y-components of the field (each an array)
         ys: list,
             List of the x- and y-components of the astrometric residual field
             (each an array)

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
        
        q = ax.quiver(xs[0], xs[1], ys[0], ys[1], scale=1, **qdict)
        ax.quiverkey(q, 0.0, 1.8, 0.1, 'residual = 0.1 arcsec',
                     coordinates='data', labelpos='N',
                     color='darkred', labelcolor='darkred')
        
        ax.set_xlabel('[degrees]', fontsize=12);
        ax.set_ylabel('[degrees]', fontsize=12)

        ax.set_xlim(-1.95, 1.95)
        ax.set_ylim(-1.9, 2.0)
        ax.set_aspect('equal')

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
    parser.add_argument('--data',type=str,required=False,default='output/out_1.pkl',
                        help='Data file [.pkl]')
    parser.add_argument('--size',type=int,required=False,default=10,
                        help='Size of training sample [int]')
    args = parser.parse_args()

    # load an example data output:
    out = pickle.load(open(args.data, 'rb'))

    #---------------------------------------------------------------------------
    
    af = AstrometryField(out)
    
    xs0_train, xs1_train, ys0_train, ys1_train, xs0_test, xs1_test, ys0_test, ys1_test = af.get_train_test(size=args.size)

    #xs = [af.thx, af.thy]
    #ys = [af.res_x, af.res_y]
    xs = [xs0_train, xs1_train]
    ys = [ys0_train, ys1_train]

    #---------------------------------------------------------------------------

    # Sanity check plot: 2-point correlation function for a given subsample
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    af.plot_astrometric_residuals(axs[0], xs, ys)
    af.plot_2pcf(axs[1], xs, ys, bins=10)

    plt.suptitle(f'{os.path.basename(args.data)}')
    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------------------------

    # Sanity check plot: Distribution of pair products of a given subsample

    # Compute the separations between each point in the field
    seps = af.compute_separations(xs)

    # Define distance bins across the range of separation values
    rs = np.histogram_bin_edges(seps, bins=10)
    bins = np.array([rs[0:-1], rs[1:]]).T

    # For each bin, select the astrometric-residual values of the points
    # that lie within that distance bin
    vals = [af.select_values(xs, ys, b) for b in bins]

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    #axs[0].hist([np.outer(val[:,0], val[:,0])[~np.tri(len(val),k=-1,dtype=bool)] for val in vals], bins=50, histtype='step')
    #axs[0].vlines([np.nanmean(np.outer(val[:,0], val[:,0])[~np.tri(len(val),k=-1,dtype=bool)]) for val in vals], ymin=0, ymax=1)
    #axs[0].set_xlabel(r'${\rm res}_{x, i} \times {\rm res}_{x, j}$')
    axs[0].axhline(y=0, c='gray', ls='--')
    pps0 = [np.outer(val[:,0], val[:,0])[~np.tri(len(val),k=-1,dtype=bool)] for val in vals]
    pps0 = [pp[np.isfinite(pp)] for pp in pps0]
    axs[0].violinplot(pps0,
                      showextrema=False,
                      showmeans=True)
    axs[0].set_xlabel(r'Distance bin')
    axs[0].set_ylabel(r'${\rm res}_{x, i} \times {\rm res}_{x, j}$')
    #axs[0].set_ylim(-0.0001, 0.0001)

    #axs[1].hist([np.outer(val[:,1], val[:,1])[~np.tri(len(val),k=-1,dtype=bool)] for val in vals], bins=50, histtype='step')
    #axs[1].vlines([np.nanmean(np.outer(val[:,1], val[:,1])[~np.tri(len(val),k=-1,dtype=bool)]) for val in vals], ymin=0, ymax=1)
    #axs[1].set_xlabel(r'${\rm res}_{y, i} \times {\rm res}_{y, j}$')
    axs[1].axhline(y=0, c='gray', ls='--')
    pps1 = [np.outer(val[:,1], val[:,1])[~np.tri(len(val),k=-1,dtype=bool)] for val in vals]
    pps1 = [pp[np.isfinite(pp)] for pp in pps0]
    axs[1].violinplot(pps1,
                      showextrema=False,
                      showmeans=True)
    axs[1].set_xlabel(r'Distance bin')
    axs[1].set_ylabel(r'${\rm res}_{y, i} \times {\rm res}_{y, j}$')
    #axs[1].set_ylim(-0.0001, 0.0001)

    plt.suptitle(f'{os.path.basename(args.data)}')
    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------------------------

    # Sanity check plot: Distribution of 2-point correlation function over
    #                    many subsamples

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    af.plot_astrometric_residuals(axs[0], [af.thx, af.thy], [af.res_x, af.res_y])

    N = 100
    rs = np.linspace(0, 4, 11)
    bins = np.array([rs[0:-1], rs[1:]]).T
    pcfs0 = np.empty((N, len(bins)))
    pcfs1 = np.empty((N, len(bins)))

    for i in range(N):
        xs0_train, xs1_train, ys0_train, ys1_train, xs0_test, xs1_test, ys0_test, ys1_test = af.get_train_test(size=args.size)
        xs = [xs0_train, xs1_train]
        ys = [ys0_train, ys1_train]
        seps = af.compute_separations(xs)
        vals = [af.select_values(xs, ys, b) for b in bins]
        pcfs = np.array([af.compute_2pcf(val) for val in vals])
        pcfs0[i] = pcfs[:,0]
        pcfs1[i] = pcfs[:,1]

    axs[1].violinplot(dataset=pcfs0, positions=0.5*(bins[:,0]+bins[:,1]))
    axs[1].set_xlabel(r'$\Delta$', fontsize=12);
    axs[1].set_ylabel(r'$\xi(\Delta)$', fontsize=12)
    axs[1].set_title(r'$\xi_{xx}$')

    axs[2].violinplot(dataset=pcfs1, positions=0.5*(bins[:,0]+bins[:,1]))
    axs[2].set_xlabel(r'$\Delta$', fontsize=12);
    axs[2].set_ylabel(r'$\xi(\Delta)$', fontsize=12)
    axs[2].set_title(r'$\xi_{yy}$')

    plt.suptitle(f'{os.path.basename(args.data)}')
    plt.tight_layout()
    plt.show()

    #---------------------------------------------------------------------------

    #m = af.gpr_train([xs0_train, xs1_train], [ys0_train, ys1_train])
    ##Xs0, Xs1 = np.meshgrid(xs0_train, xs1_train)
    ##Xs0, Xs1 = np.meshgrid(xs0_test, xs1_test)
    #Xs0, Xs1 = np.meshgrid(np.linspace(np.min(af.thx), np.max(af.thx), 100),
    #                       np.linspace(np.min(af.thy), np.max(af.thy), 100))
    #pred, var = af.gpr_predict(m, [Xs0, Xs1], [ys0_test, ys1_test])
    #
    #fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    #af.plot_astrometric_residuals(axs[0], [af.thx, af.thy], [af.res_x, af.res_y])
    #af.plot_astrometric_residuals(axs[1], [Xs0, Xs1], [pred[:,:,0], pred[:,:,1]])
    #
    #plt.show()
