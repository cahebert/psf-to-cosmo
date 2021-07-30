## TO DO LIST:
## Switch to xip/xim
## put generic photo-z bin
## remove stuff specific to HSC

import numpy as np
import pylab as plt
import pyccl as ccl
import copy
import warnings


def gauss_photo_z(z, z0, sigma_z):
    """
    Define gaussian photo-z distribution.

    :z:       array. Redshift range.
    :z0:      float. Central position of the redshift bin.
    :sigma_z: float. Width of the redshift distribution.
    """
    return 3 * np.exp(-0.5 * (z - z0) ** 2 / sigma_z ** 2)


class comp_shear_cl(object):

    def __init__(self, Omega_ch2=0.1, Omega_bh2=0.023, AS=None, S8=None,
                 Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1, alpha=0.45,
                 matter_power_spectrum='halofit', A0=0, eta=0,
                 delta_m=0., delta_z=[0., 0., 0., 0.],
                 ell=np.arange(20, 3000)):

        self._matter_power_spectrum = matter_power_spectrum

        self.update_mbias(delta_m=delta_m)
        self.update_IA(A0=A0, eta=eta)
        self.update_cosmology(Omega_ch2=Omega_ch2, Omega_bh2=Omega_bh2,
                              AS=AS, S8=S8, alpha=alpha,
                              Omega_nu_h2=Omega_nu_h2,
                              H0=H0, ns=ns, w0=w0)
        self.load_photo_z()
        self.update_redshift_bias(delta_z)

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

        self._Omega_b = self.Omega_bh2 / self._h ** 2
        self._Omega_c = self.Omega_ch2 / self._h ** 2
        self._Omega_m = self._Omega_b + self._Omega_c
        self.Omega_nu_h2 = Omega_nu_h2
        self._m_nu = (self.Omega_nu_h2 / self._h ** 2) * 93.14

        if S8 is None and AS is None:
            raise ValueError('S8 or AS should be given')
        if S8 is not None and AS is not None:
            raise ValueError('Just S8 or AS should be given')

        if S8 is not None:
            self.AS = None
            self._A_s = None
            self._sigma8 = S8 * (1. / (self._Omega_m / 0.3) ** alpha)
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

    def load_photo_z(self, z0=[0.3, 0.6, 0.8, 1.1], plot=True):

        z = np.linspace(0, 3, 200)
        self.redshifts = []
        self.nz = []

        for i in range(len(z0)):
            self.redshifts.append(z)
            self.nz.append(0.02 * gauss_photo_z(z, z0[i], 0.1))

        self.redshifts = np.array(self.redshifts)
        self.nz = np.array(self.nz)

        if plot:
            C = ['k', 'b', 'y', 'r']
            plt.figure(figsize=(12, 4))
            for i in range(4):
                plt.plot(self.redshifts[i], self.nz[i], C[i], lw=3)

            plt.plot([0, 2.6], [0, 0], 'k--')
            plt.xlim(0, 2.6)
            plt.ylim(-0.1, 0.2)
            plt.show()

    def intrinsic_al(self, redshift, A0=1, eta=1, z0=0.62):
        AI = A0 * ((1 + redshift) / (1 + z0)) ** eta
        return AI

    def multiplicatif_bias(self, Bin_i, Bin_j, delta_m=0.01 * 100.):

        dm = delta_m / 100.
        m_sel = np.array([0.86, 0.99, 0.91, 0.91]) / 100.
        m_R = np.array([0, 0, 1.5, 3.0]) / 100.

        return (1 + dm) ** 2 * (1 + m_sel[Bin_i] + m_R[Bin_i]) * (1 + m_sel[Bin_j] + m_R[Bin_j])

    def comp_Cl(self):
        self.wl_bin_shear = []
        nell = len(self.ell)
        self._Cl = np.zeros(10 * nell)

        self.Cl = np.zeros(len(self.ell), dtype={'names': ('11', '12', '13', '14',
                                                           '22', '23', '24',
                                                           '33', '34',
                                                           '44'),
                                                 'formats': ('f8', 'f8', 'f8', 'f8',
                                                             'f8', 'f8', 'f8',
                                                             'f8', 'f8',
                                                             'f8')})
        for i in range(4):
            filtre = (self.redshifts_bias[i] > 0)
            # je ne sais plus si c'est important que le redshift de l ai soit le meme que
            # celui de la distribution des galaxies 
            # AI = self.intrinsic_al(self.redshifts[i][filtre], A0=self.A0, eta=self.eta, z0=0.62)
            AI = self.intrinsic_al(self.redshifts[i], A0=self.A0, eta=self.eta, z0=0.62)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.wl_bin_shear.append(ccl.WeakLensingTracer(self.cosmology,
                                                               dndz=(
                                                                   self.redshifts_bias[i][filtre], self.nz[i][filtre]),
                                                               has_shear=True,
                                                               ia_bias=(self.redshifts[i], AI)))
                # ia_bias=(self.redshifts[i][filtre], AI)))
        I = 0
        for i in range(4):
            for j in range(4):
                if j >= i:
                    key = "%i%i" % ((i + 1, j + 1))

                    cl = ccl.angular_cl(self.cosmology, self.wl_bin_shear[i], self.wl_bin_shear[j], self.ell)

                    m_ij = self.multiplicatif_bias(i, j, delta_m=self.delta_m)
                    cl *= m_ij
                    self.Cl[key] = cl
                    self._Cl[I * nell:(I + 1) * nell] = self.Cl[key]
                    I += 1


if __name__ == '__main__':
    csc = comp_shear_cl(Omega_ch2=0.1, Omega_bh2=0.023, AS=4.,
                        Omega_nu_h2=1e-3, H0=70, ns=0.97, w0=-1,
                        matter_power_spectrum='halofit',
                        ell=np.arange(20, 3000))
    csc.comp_Cl()
