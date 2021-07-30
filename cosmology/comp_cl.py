## TO DO LIST:
## Switch to xip/xim
## put generic photo-z bin
## remove stuff specific to HSC

import numpy as np
import pylab as plt
import pyccl as ccl
import copy
from astropy.io import fits
import warnings
import shear_subaru
import os
from scipy.stats import binned_statistic
path = os.path.dirname(shear_subaru.__file__)

def rebinning_log(ell, Cl, lmin=62.33783875, lmax=6725.85492173, nbins=16):
    """
    Rebinning of the angular power spectrum in log bin.

    :ell:    array. Multipole l.
    :Cl:     array. Angular power spectrum.
    :lmin:   float. Minimum multipole l. (default: lmin=25)
    :lmax:   float. Maximum multipole l. (default: lmin=2025)
    :nbins:  int. Number of bin in the new binning. (default: nbins=15)
    """
    Filtre = [False, False, False, False, False, 
              True, True, True, True, True, True,
              False, False, False, False]
    
    BINNING = 10 ** np.linspace(np.log10(lmin), np.log10(lmax), nbins)

    weight =  ell**2 / (2. * np.pi)
    sum_wp, B, C = binned_statistic(ell, Cl*weight, statistic='sum', bins=BINNING, range=None)
    sum_w, B, C = binned_statistic(ell, weight, statistic='sum', bins=BINNING, range=None)
    Cl_bin = sum_wp[Filtre] / sum_w[Filtre]

    log_B = np.log10(BINNING)
    log_B = log_B[:-1] + (log_B[1] - log_B[0])/2.
    ell_bin = 10**log_B
    
    l_errm = ell_bin - BINNING[:-1] 
    l_errp = BINNING[1:] - ell_bin
    
    l_err = [l_errm[Filtre], l_errp[Filtre]]

    return Cl_bin, ell_bin[Filtre], l_err

def return_llplus1(ell, Cl, cst=2.*np.pi):
    return (ell * (ell + 1) * Cl) / cst

class comp_shear_cl(object):

    def __init__(self, Omega_ch2=0.1, Omega_bh2=0.023, AS=None, S8=None,
                 Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, alpha=0.45,
                 matter_power_spectrum = 'halofit', A0=0, eta=0,
                 delta_m = 0., delta_z=[0., 0., 0., 0.],
                 alpha_psf = 0.057, beta_psf=-1.22,
                 ell = np.arange(20, 3000), photo_z_method='pf',
                 log_binning=False):

        self._matter_power_spectrum = matter_power_spectrum
        self.photo_z_method = photo_z_method
        self.log_binning=log_binning
        if self.log_binning:
            print('log binning pf')
        else:
            print('binning paper')

        self.update_mbias(delta_m=delta_m)
        self.update_IA(A0=A0, eta=eta)
        self.update_cosmology(Omega_ch2=Omega_ch2, Omega_bh2=Omega_bh2,
                              AS=AS, S8=S8, alpha=alpha,
                              Omega_nu_h2=Omega_nu_h2,
                              H0=H0, ns=ns, w0=w0)
        self.load_photo_z(photo_z_method=self.photo_z_method)
        self.update_redshift_bias(delta_z)

        print('psf leakage used')
        self.load_psf_leakage()
        self.update_psf_leakage(alpha_psf=alpha_psf,
                                beta_psf=beta_psf)

        self.ell = ell

    def update_mbias(self, delta_m=0):
        self.delta_m = delta_m

    def update_IA(self, A0=1, eta=1):
        self.A0 = A0
        self.eta = eta

    def update_redshift_bias(self, delta_z=[0., 0., 0., 0.]):
        self.delta_z = delta_z
        dz = copy.deepcopy(np.array(delta_z) / 100.)
        self.redshifts_bias = []
        for i in range(len(dz)):
            new_z = self.redshifts[i] - dz[i]
            self.redshifts_bias.append(copy.deepcopy(self.redshifts[i]) - dz[i])

    def update_cosmology(self, Omega_ch2=0.1, Omega_bh2=0.023, AS=None, S8=None,
                         Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, alpha=0.45):

        self.Omega_ch2 = Omega_ch2
        self.Omega_bh2 = Omega_bh2
        self.H0 = H0
        self._h = self.H0 / 100.

        self._Omega_b = self.Omega_bh2 / self._h**2
        self._Omega_c = self.Omega_ch2 / self._h**2
        self._Omega_m = self._Omega_b + self._Omega_c
        self.Omega_nu_h2 = Omega_nu_h2
        self._m_nu = (self.Omega_nu_h2 / self._h**2) * 93.14

        if S8 is None and AS is None:
            raise ValueError('S8 or AS should be given')
        if S8 is not None and AS is not None:
            raise ValueError('Just S8 or AS should be given')

        if S8 is not None:
            self.AS = None
            self._A_s = None
            self._sigma8 = S8 * (1./ (self._Omega_m/0.3)**alpha)
        if AS is not None:
            self.AS = AS
            self._A_s = np.exp(self.AS) * 1e-10
            self._sigma8 = None

        self.n_s = ns
        self.w0 = w0

        self.cosmology = ccl.Cosmology(Omega_c=self._Omega_c, Omega_b=self._Omega_b,
                                       h=self._h, n_s=self.n_s, sigma8=self._sigma8, A_s=self._A_s,
                                       w0=self.w0, m_nu=self._m_nu,
                                       matter_power_spectrum=self._matter_power_spectrum)

    def load_photo_z(self, photo_z_method='pf', plot=False):

        if photo_z_method not in ['pf', 'hamana']:
            raise ValueError('photo_z_method must be pf or hamana')
        print('Photo-z: %s is used.'%(photo_z_method))
        if photo_z_method == 'pf':
            key = ''
        if photo_z_method == 'hamana':
            key = '_hamana'

        self.redshifts = []
        self.nz = []

        for i in range(4):
            file_bin = np.loadtxt(os.path.join(path, 'data/photo-z/bin%i'%(i+1))+key+'.dat', comments='#')
            self.redshifts.append(file_bin[:,0])
            self.nz.append(file_bin[:,1])

        self.redshifts = np.array(self.redshifts)
        self.nz = np.array(self.nz)

        if plot:
            C = ['k', 'b', 'y', 'r']
            plt.figure(figsize=(12,4))
            for i in range(4):
                plt.plot(self.redshifts[i], self.nz[i], C[i], lw=3)

            plt.plot([0,2.6], [0,0], 'k--')
            plt.xlim(0,2.6)
            plt.ylim(-0.1,3.8)
            plt.show()


    def load_psf_leakage(self, alpha_psf = 0.057, beta_psf=-1.22):

        psf = np.loadtxt(os.path.join(path, 'data/psf/pk_psfleak_and_res.txt'))
        mask = [False, True, True, True, True, True, True, False, False]
        self.Cl_pp = psf[:,1][mask] / (alpha_psf * alpha_psf)
        self.Cl_pq = psf[:,2][mask] / (alpha_psf * beta_psf)
        self.Cl_qq = psf[:,3][mask] / (beta_psf * beta_psf)

    def update_psf_leakage(self, alpha_psf = 0.057, beta_psf=-1.22):

        self.alpha_psf = alpha_psf
        self.beta_psf = beta_psf
        self.Cl_psf = self.alpha_psf**2 * self.Cl_pp
        self.Cl_psf += self.alpha_psf * self.beta_psf * self.Cl_pq
        self.Cl_psf += self.beta_psf**2 *self.Cl_qq

    def intrinsic_al(self, redshift, A0=1, eta=1, z0=0.62):
        AI = A0 * ((1+redshift) / (1+z0))**eta
        return AI

    def multiplicatif_bias(self, Bin_i, Bin_j, delta_m=0.01*100.):

        dm = delta_m / 100.
        m_sel = np.array([0.86, 0.99, 0.91, 0.91]) / 100.
        m_R = np.array([0, 0, 1.5, 3.0]) / 100.

        return (1+dm)**2 * (1+m_sel[Bin_i]+m_R[Bin_i]) * (1+m_sel[Bin_j]+m_R[Bin_j])

    def comp_Cl(self):
        self.wl_bin_shear = []
        nell = len(self.ell)
        self._Cl = np.zeros(10*nell)

        self.Cl = np.zeros(len(self.ell), dtype={'names':('11', '12', '13', '14',
                                                          '22', '23', '24',
                                                          '33', '34',
                                                          '44'),
                                                 'formats':('f8', 'f8', 'f8', 'f8',
                                                            'f8', 'f8','f8',
                                                            'f8', 'f8',
                                                            'f8')})
        for i in range(4):
            filtre = (self.redshifts_bias[i] > 0)
            # je ne sais plus si c'est important que le redshift de l ai soit le meme que
            # celui de la distribution des galaxies 
            #AI = self.intrinsic_al(self.redshifts[i][filtre], A0=self.A0, eta=self.eta, z0=0.62)
            AI = self.intrinsic_al(self.redshifts[i], A0=self.A0, eta=self.eta, z0=0.62)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.wl_bin_shear.append(ccl.WeakLensingTracer(self.cosmology,
                                                               dndz=(self.redshifts_bias[i][filtre], self.nz[i][filtre]),
                                                               has_shear=True,
                                                               ia_bias=(self.redshifts[i], AI)))
                                                               #ia_bias=(self.redshifts[i][filtre], AI)))
        I = 0
        for i in range(4):
            for j in range(4):
                if j>=i:
                    key = "%i%i"%((i+1, j+1))
                    if self.log_binning:
                        #ell = np.arange(60, 6501).astype(float)
                        # approx permet de gagner un facteur 10 en
                        # temps de calcul sans changer la precision
                        # sur les Cl
                        ell = np.linspace(306, 1977, 400)
                        mask_scale = ((ell >= 306) & (ell <= 1977))
                        ell = ell[mask_scale]
                        clb = ccl.angular_cl(self.cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], ell)
                        cl, ell_bin, ell_err = rebinning_log(ell, clb, lmin=62.33783875, lmax=6725.85492173, nbins=16)
                    else:
                        cl = ccl.angular_cl(self.cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], self.ell)

                    m_ij = self.multiplicatif_bias(i, j, delta_m=self.delta_m)
                    cl *= m_ij
                    self.Cl[key] = return_llplus1(self.ell, cl, cst=2.*np.pi) + self.Cl_psf
                    self._Cl[I*nell:(I+1)*nell] = self.Cl[key]
                    I += 1

if __name__ == '__main__':

    csc = comp_shear_cl(Omega_ch2=0.1, Omega_bh2=0.023, AS=4.,
                        Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1,
                        matter_power_spectrum = 'halofit',
                        ell=np.arange(20, 3000))
    csc.comp_Cl()