"""
Module containing the AstrometryField class for analyzing astrometric
residual fields.
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import treegp

def load_data(infile: str) -> dict:
    """
    Load astrometric field data from file.

    Parameters
    ----------
    infile:
        Input file to analyze

    Returns
    -------
    ddict:
        Dictionary of astrometric field data
    """
    with open(infile, 'rb') as data:
        ddict = pickle.load(data)
    return ddict

def compute_2pcf(xs: np.ndarray,
                 ys: np.ndarray,
                 bins: int = 10) -> tuple[np.ndarray]:
    """
    Compute the 2-point correlation function of each component of the
    astrometric residual field.

    Parameters
    ----------
    xs:
        Array of the x- and y-components of the field
    ys:
        Array of the x- and y-components of the astrometric residual field
    bins:
        The number of distance bins at which to compute the 2-point
        correlation functions

    Returns
    -------
    dr:
        separations at which the 2-point correlation functions were
        calculated
    xi0:
        2-point correlation function of the x-component of the astrometric
        residual field
    xi1:
        2-point correlation function of the y-component of the astrometric
        residual field
    """
    # ``upper triangle`` indices for an (N, N) array
    ind = np.triu_indices(xs.shape[0])

    # seps has shape (N, N) up to ind
    # calculate the euclidean separation between each point in the field
    seps = np.hypot((xs[:, 0] - xs[:, 0, np.newaxis])[ind],
                    (xs[:, 1] - xs[:, 1, np.newaxis])[ind])
    # seps = distance_matrix(xs, xs)[ind]

    # pps0, pps1 have shape (N, N) up to ind
    # calculate the pair products of each component of each point of the
    # astrometric residuals
    pps0 = np.outer(ys[:, 0], ys[:, 0])[ind]
    pps1 = np.outer(ys[:, 1], ys[:, 1])[ind]

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


def plot_2pcf(ax: matplotlib.axes.Axes,
              dr: np.ndarray,
              xi0: np.ndarray,
              xi1: np.ndarray) -> None:
    """
    Plot the two-point correlation function for the x- and y-components of
    the astrometric residual field as a function of distance between
    points.

    Parameters
    ----------
    ax:
        Matplotlib axis in which to plot
    dr:
        separations at which the 2-point correlation functions were
        calculated
    xi0:
        2-point correlation function of the x-component of the astrometric
        residual field
    xi1:
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
    ax.set_xlabel(r'$\Delta$ [degrees]', fontsize=12)
    ax.set_ylabel(r'$\xi(\Delta)$ [degrees$^2$]', fontsize=12)


def plot_astrometric_residuals(ax: matplotlib.axes.Axes,
                               xs: np.ndarray,
                               ys: np.ndarray) -> None:
    """
    Plot the astrometric residual field of a set of points.

    Parameters
    ----------
    ax:
        Matplotlib axis in which to plot
    xs:
        Array of the x- and y-components of the field
    ys:
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

    q = ax.quiver(xs[:, 0], xs[:, 1], ys[:, 0], ys[:, 1], scale=1, **qdict)
    ax.quiverkey(q, 0.0, 1.8, 0.1, 'residual = 0.1 arcsec',
                 coordinates='data', labelpos='N',
                 color='darkred', labelcolor='darkred')

    ax.set_xlabel('RA [degrees]')
    ax.set_ylabel('Dec [degrees]')

    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.9, 2.0)
    ax.set_aspect('equal')


class AstrometryField:
    """
    Astrometry field class that contains methods for loading in
    simulated data, computing mean fields, and interpolating over
    positions with Gaussian process methods.
    """
    def __init__(self, infile: str, pscale: float = 0.2, bins: int = 10):
        """
        Initialize AstrometryField class.

        Parameters
        ----------
        infile:
            Dictionary output from psfws
        pscale:
            Pixel scale in asec (0.2 for LSST)
        bins:
            The number of distance bins at which to compute the 2-point
            correlation functions

        Returns
        -------
        AstrometryField
        """
        self.ddict = load_data(infile)

        #self.x = np.array(self.ddict['x'])
        #self.y = np.array(self.ddict['y'])
        #self.thx = np.array(self.ddict['thx'])
        #self.thy = np.array(self.ddict['thy'])

        self._pscale = pscale
        res_x, res_y = self.get_astrometric_residuals()

        self.xs = np.asarray([self.ddict['thx'], self.ddict['thy']]).T
        self.ys = np.asarray([res_x, res_y]).T

        self._bins = bins

        self.gp_x = None
        self.gp_y = None

    def get_astrometric_residuals(self) -> tuple[np.ndarray]:
        """
        Compute the x- and y-components of the astrometric residuals by
        subtracting the mean of each field, respectively.

        Parameters
        ----------
        None

        Returns
        -------
        res_x:
            x-component of the astrometric residual field
        res_y:
            y-component of the astrometric residual field
        """
        # The measurements are done per psf by simulating a 50x50 pixel image
        # on the sky centered at some location thx, thy.
        #
        # The centroid (x,y) is measured in pixels relative to bottom corner of
        # this image. To get the residuals, subtract 25 pixels from the x,y
        # values to shift the origin of this 50x50 image to thx, thy; that's
        # it! Now x and y are the residuals. Convert from pixels to angular
        # distance using the pixel scale: here using the LSST pixel scale 0.2
        # to get units of arcseconds.

        # astrometric residuals:
        res_x = np.asarray([(x-25) * self._pscale for x in self.ddict['x']])
        res_y = np.asarray([(y-25) * self._pscale for y in self.ddict['y']])

        # subtract 1st order polynomial (i.e., mean):
        # according to P-F this should be a third order polynomial! to-do.
        res_x = np.asarray([x-np.mean(res_x) for x in res_x])
        res_y = np.asarray([y-np.mean(res_y) for y in res_y])

        return res_x, res_y

    def get_train_test(self, size: int) -> tuple[np.ndarray]:
        """
        Split the astrometric residual field into training and testing subsets.

        Parameters
        ----------
        size: int,
            Size (number) of points in the training subset.

        Returns
        -------
        xs_train:
            Training subset of the field components
        xs_test:
            Testing subset of the field components
        ys_train:
            Training subset of the astrometric residual components
        ys_test:
             Testing subset of the astrometric residual components
        """
        return train_test_split(self.xs,
                                self.ys,
                                train_size=size,
                                random_state=42)

    def initialize_gp(self, xs_train, ys_train):
        kernel = "15**2 * AnisotropicVonKarman(invLam=np.array([[1./3000.**2,0],[0,1./3000.**2]]))"
        gp_x = treegp.GPInterpolation(kernel=kernel,
                                      optimizer='anisotropic', 
                                      normalize=False,
                                      white_noise=0.,
                                      p0=[3000., 0.,0.],
                                      n_neighbors=4,
                                      average_fits=None,
                                      nbins=20, 
                                      min_sep=None,
                                      max_sep=None)
        gp_y = treegp.GPInterpolation(kernel=kernel,
                                      optimizer='anisotropic', 
                                      normalize=False,
                                      white_noise=0.,
                                      p0=[3000., 0.,0.],
                                      n_neighbors=4,
                                      average_fits=None,
                                      nbins=20, 
                                      min_sep=None,
                                      max_sep=None)

        gp_x.initialize(xs_train, ys_train[:, 0], y_err=None)
        gp_y.initialize(xs_train, ys_train[:, 1], y_err=None)
        
        self.gp_x = gp_x
        self.gp_y = gp_y

    def predict(self, xs_test):
        ys_predict_x, ys_cov_x = self.gp_x.predict(xs_test, return_cov=True)
        ys_predict_y, ys_cov_y = self.gp_y.predict(xs_test, return_cov=True)
        ys_pred = np.asarray([ys_predict_x, ys_predict_y]).T
        ys_cov = np.asarray([ys_cov_x, ys_cov_y]).T
        
        return ys_pred, ys_cov



if __name__ == '__main__':
    # Main function for testing and debugging purposes.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile',
                        type=str,
                        required=False,
                        default='output/out_1.pkl',
                        help='Input data file [.pkl]')
    parser.add_argument('--size',
                        type=int,
                        required=False,
                        default=100,
                        help='Size of training sample [int]')
    args = parser.parse_args()

    af = AstrometryField(args.infile)

    xs_train, xs_test, ys_train, ys_test = af.get_train_test(size=args.size)

    # We'll use independent GPs to model the x- and y-components of the
    # astrometric residual field. These can be mixed if we decide it's
    # important at some later point.

    af.initialize_gp(xs_train, ys_train)

    ys_pred, ys_cov = af.predict(xs_test)

    fig, axs = plt.subplots(2, 2, figsize=(16, 8))

    plot_astrometric_residuals(axs[0,0], xs_test, ys_test)
    plot_astrometric_residuals(axs[0,1], xs_test, ys_pred)

    sc = axs[1,0].scatter(xs_test[:,0], xs_test[:,1], c=(ys_test-ys_pred)[:,0], cmap='seismic', s=1, vmin=-0.05, vmax=0.05)
    axs[1,0].set_xlabel('RA [degrees]', fontsize=12);
    axs[1,0].set_ylabel('Dec [degrees]', fontsize=12)
    axs[1,0].set_xlim(-1.95, 1.95)
    axs[1,0].set_ylim(-1.9, 2.0)
    axs[1,0].set_aspect('equal')
    divider = make_axes_locatable(axs[1,0])
    cax = divider.append_axes('right', size = '5%', pad=0)
    plt.colorbar(sc, cax=cax)

    sc = axs[1,1].scatter(xs_test[:,0], xs_test[:,1], c=(ys_test-ys_pred)[:,1], cmap='seismic', s=1, vmin=-0.05, vmax=0.05)
    axs[1,1].set_xlabel('RA [degrees]', fontsize=12);
    axs[1,1].set_ylabel('Dec [degrees]', fontsize=12)
    axs[1,1].set_xlim(-1.95, 1.95)
    axs[1,1].set_ylim(-1.9, 2.0)
    axs[1,1].set_aspect('equal')
    divider = make_axes_locatable(axs[1,1])
    cax = divider.append_axes('right', size = '5%', pad=0)
    plt.colorbar(sc, cax=cax)

    # cb = fig.colorbar(sc, ax=axs[1,:2], location='bottom', shrink=0.6)

    axs[0,0].set_title('Expected')
    axs[0,1].set_title('Predicted')

    axs[1,0].set_title('Expected - Predicted (x)')
    axs[1,1].set_title('Expected - Predicted (y)')

    plt.tight_layout()
    plt.show()
