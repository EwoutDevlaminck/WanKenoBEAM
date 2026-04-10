"""Model for the fluctuation envelope amplitude.

The fluctuation amplitude is factored into a radial profile (read from
nefluct.dat in the same directory as this file) and a poloidal ballooning
envelope:

    delta_ne/ne(rho, theta) = A_rho(rho) * A_theta(theta)

where

    A_theta(theta) = 0.5 * (1 + ballooning + (1 - ballooning) * cos(theta))

    ballooning=1  →  isotropic  (A_theta = 1 everywhere)
    ballooning=0  →  fully ballooned  (A_theta = cos^2(theta/2))
"""

import os
import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Load the fluctuation profile from nefluct.dat (same directory as this file)
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_Ne_fluct = np.loadtxt(os.path.join(_this_dir, 'nefluct.dat'), skiprows=1)
rho_fl = _Ne_fluct[:, 0]
Fluct  = _Ne_fluct[:, 1]

# Extend the rho grid to rho = 3 so the interpolant is defined outside [0, 1]
_drho      = rho_fl[1] - rho_fl[0]
_extendrho = np.arange(rho_fl[-1] + _drho, 3.0, _drho)
rho_fl = np.concatenate((rho_fl, _extendrho))
Fluct  = np.concatenate((Fluct,  Fluct[-1] * np.ones(len(_extendrho))))

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

# Ballooning parameter: 1 = isotropic, 0 = fully ballooned
ballooning = 0.5

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

_ampl_rho_spline = interp1d(rho_fl, Fluct, kind='cubic')

def _ampl_rho(rho):
    return _ampl_rho_spline(rho)

def _ampl_theta(theta):
    return 0.5 * (1 + ballooning + (1 - ballooning) * np.cos(theta))


def scatteringDeltaneOverne(ne, rho, theta):
    """Return delta_ne/ne at (rho, theta).  Called by WKBeam for each ray point."""
    return _ampl_rho(rho) * _ampl_theta(theta)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Ne_file = np.loadtxt(os.path.join(_this_dir, 'ne.dat'), skiprows=1)
    rho_ne, Ne = Ne_file[:, 0], Ne_file[:, 1]

    theta = 0.0
    delt  = scatteringDeltaneOverne(Ne, rho_ne, theta)

    plt.figure()
    plt.plot(rho_ne, delt, label=f'ballooning = {ballooning}')
    plt.grid()
    plt.xlim(0, 1.1)
    plt.ylim(0, max(delt) * 1.2)
    plt.xlabel(r'$\rho_\psi$', fontsize=14)
    plt.ylabel(r'$\mathrm{RMS}\ \delta n_e / n_e$', fontsize=14)
    plt.title('Fluctuation envelope amplitude  (theta = 0)')
    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.show()
