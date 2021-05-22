"""
Module containing the AstrometryField class for analyzing astrometric residual fields.
"""
import os
import pickle
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn
seaborn.set_palette('bright')

import treecorr

#-------------------------------------------------------------------------------

# see discussion here: https://stackoverflow.com/questions/50185399/multiple-output-gaussian-process-regression-in-scikit-learn

import tensorflow as tf
import gpflow as gpf

from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter

MAXITER = ci_niter(5000)

#-------------------------------------------------------------------------------

def optimize_model_with_scipy(model, data):
    """
    From https://gpflow.readthedocs.io/en/master/notebooks/advanced/multioutput.html
    """
    optimizer = gpf.optimizers.Scipy()
    optimizer.minimize(
        model.training_loss_closure(data),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        options={"disp": True, "maxiter": MAXITER}
    )

# Follow https://gpflow.readthedocs.io/en/master/notebooks/advanced/natural_gradients.html
# Empirically, want to keep gamma ~ 0.1--1.0, and learning_rate ~ 0.1--1.0
# This might change depending on the number of inducing points
# Optimize the model over the data
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
adam_opt_for_svgp = tf.optimizers.Adam(learning_rate=0.01)

# Need to decorate this step following https://stackoverflow.com/a/61864311
@tf.function
def optimization_step(model, data):
    natgrad_opt.minimize(model.training_loss_closure(data), var_list=[(model.q_mu, model.q_sqrt)])
    adam_opt_for_svgp.minimize(model.training_loss_closure(data), var_list=model.trainable_variables)

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
                                                                train_size=size,
                                                                random_state=42)

        return xs_train, xs_test, ys_train, ys_test

    def compute_2pcf(self, xs, ys, bins=10):
        """
        Compute the 2-point correlation function of each component of the
        astrometric residual field.

        Parameters
        ----------
         xs: ndarray,
             Array of the x- and y-components of the field
         ys: ndarray,
             Array of the x- and y-components of the astrometric residual field
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
        # ``upper triangle`` indices for an (N, N) array
        ind = np.triu_indices(xs.shape[0])

        # seps has shape (N, N) up to ind
        # calculate the euclidean separation between each point in the field
        #seps = np.hypot((xs[:,0] - xs[:,0,np.newaxis])[ind], (xs[:,1] - xs[:,1,np.newaxis])[ind])
        seps = distance_matrix(xs, xs)[ind]

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

        # Normalize quantities
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
             Array of the x- and y-components of the field
         ys: ndarray,
             Array of the x- and y-components of the astrometric residual field

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
             Array of the x- and y-components of the field
         ys: ndarray,
             Array of the x- and y-components of the astrometric residual field
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


    def gpr_train(self, xs, ys):
        """
        Perform Gaussian Process regression over the astrometric residual field

        Parameters
        ----------
         xs: ndarray,
             Array of the x- and y-components of the field
         ys: ndarray,
             Array of the x- and y-components of the astrometric residual field

        Returns
        -------
         m: ???,
             `GPy` model
        """
        k = GPy.kern.RBF(input_dim=2, active_dims=[0,1], ARD=True, variance=0.1, lengthscale=[0.01, 0.01]) + GPy.kern.White(input_dim=2)
        
        m = GPy.models.GPRegression(xs, ys, kernel=k)
        
        m.optimize(messages=True)

        return m

    def gpr_predict(self, m, xs, ys):
        """
        Perform Gaussian Process regression over the astrometric residual field

        Parameters
        ----------
         m: ???,
             `GPy` model
         xs: ndarray,
             Array of the x- and y-components of the field

        Returns
        -------
         pred: ndarray,
             Gaussian Process prediction mean over xs
         var: ndarray,
             Gaussian Process prediction variance over xs
        """
        pred, var = m.predict(xs)

        return pred, var


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

    xs_grid = np.array([x.ravel() for x in np.meshgrid(np.linspace(np.min(af.xs[:,0]), np.max(af.xs[:,0]), args.size),
                                                       np.linspace(np.min(af.xs[:,1]), np.max(af.xs[:,1]), args.size))]).T
    xs_train, xs_test, ys_train, ys_test = af.get_train_test(size=args.size)

    # Define relevant constants
    N = 5000  # number of points
    D = 2  # number of input dimensions
    M = len(xs_train)  # number of inducing points
    L = 2  # number of latent GPs
    P = 2  # number of observations = output dimensions
    #Zinit = xs_train # initialization of inducing input locations (M random points from the training inputs)
    Zinit = xs_grid # initialization of inducing input locations (M random points from the training inputs)
    Z = Zinit.copy()

    # Define the kernel and model
    k = gpf.kernels.SharedIndependent(gpf.kernels.SquaredExponential(lengthscales=0.3, variance=2e-5), output_dim=P)
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
    m = gpf.models.SVGP(k, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)

    #k_list = [gpf.kernels.SquaredExponential(lengthscales=0.3, variance=2e-5) + gpf.kernels.White() for _ in range(P)]
    #k = gpf.kernels.SeparateIndependent(k_list)
    #iv = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
    #m = gpf.models.SVGP(k, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)

    #k_list = [gpf.kernels.SquaredExponential(lengthscales=0.3, variance=2e-5) for _ in range(P)]
    #k = gpf.kernels.LinearCoregionalization(k_list, W=np.random.randn(P, L))
    #iv = gpf.inducing_variables.SharedIndependentInducingVariables(gpf.inducing_variables.InducingPoints(Z))
    ## initialize mean of variational posterior to be of shape MxL
    #q_mu = np.zeros((M, L))
    ## initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    #q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0
    #m = gpf.models.SVGP(k, gpf.likelihoods.Gaussian(), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt)

    # Stop Adam from optimizing the variational parameters
    gpf.set_trainable(m.q_mu, False)
    gpf.set_trainable(m.q_sqrt, False)

    ## Optimize the model over the data
    #optimize_model_with_scipy(m, [af.xs, af.ys])
    #gpf.utilities.print_summary(m)

    for i in range(MAXITER):
        optimization_step(m, [af.xs, af.ys])
        likelihood = m.elbo([af.xs, af.ys])
        tf.print(f"SVGP with NaturalGradient and Adam: iteration {i + 1} likelihood {likelihood:.04f}")
    gpf.utilities.print_summary(m)

    # Make predictions
    # Note: predict_f() returns denoised signal
    #       predict_y() returns noisy signal
    pred, var = m.predict_y(xs_test)

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    af.plot_astrometric_residuals(axs[0,0], af.xs, af.ys)
    af.plot_astrometric_residuals(axs[0,1], xs_train, ys_train)
    af.plot_astrometric_residuals(axs[1,0], xs_test, ys_test)
    af.plot_astrometric_residuals(axs[1,1], xs_test, pred)

    axs[0,0].set_title('Input')
    axs[0,1].set_title(f'Inducing (n = {M})')
    axs[1,0].set_title('Test')
    axs[1,1].set_title('Pred')

    plt.suptitle(f'{os.path.basename(args.infile)}')
    plt.tight_layout()
    #plt.savefig(f'{os.path.splitext(os.path.basename(args.infile))[0]}_n={M}.pdf')
    plt.show()

    #---------------------------------------------------------------------------

    #fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #k.plot(ax=ax, color='k', ls='--', label='GP kernel')
    #ax.axhline(y=0, ls='--', lw=1, c='gray')
    #ax.plot(dr, xi0, marker='o', ms=5, ls='-', lw=1, label=r'$\xi_{xx}$')
    #ax.plot(dr, xi1, marker='o', ms=5, ls='-', lw=1, label=r'$\xi_{yy}$')
    #ax.legend()
    #ax.set_xlabel(r'$\Delta$ [degrees]', fontsize=12);
    #ax.set_ylabel(r'$\xi(\Delta)$ [degrees$^2$]', fontsize=12)
    #ax.set_xlim(0, np.max(dr)+np.diff(dr)[0])
    #plt.show()

    #dr, xi0, xi1 = af.compute_2pcf(af.xs, af.ys)

    #fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    #af.plot_astrometric_residuals(axs[0], af.xs, af.ys)
    #af.plot_2pcf(axs[1], dr, xi0, xi1)
    #af.plot_astrometric_residuals(axs[0], af.xs, pred)

    #plt.suptitle(f'{os.path.basename(args.infile)}')
    #plt.tight_layout()
    #plt.show()
    

    #---------------------------------------------------------------------------

    ## Sanity check plot: 2-point correlation function
    #
    #fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    #af.plot_astrometric_residuals(axs[0], af.xs, af.ys)
    #af.plot_2pcf(axs[1], dr, xi0, xi1)

    #plt.suptitle(f'{os.path.basename(args.infile)}')
    #plt.tight_layout()
    #plt.show()

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
