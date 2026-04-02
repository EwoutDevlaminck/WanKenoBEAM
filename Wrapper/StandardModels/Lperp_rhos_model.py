"""Perpendicular correlation length from the sound Larmor radius.

The model sets

    Lperp = max(epsilon, factor * rho_s)

where rho_s is the sound Larmor radius

    rho_s = 1.02e2 * sqrt(mu * Te / Z) / Bnorm   [cm]

(Te in eV, Bnorm in Gauss, from the NRL Plasma Formulary pp. 28-29).

    mu  = m_i / m_p   (2 for deuterium)
    Z   = ion charge state
    epsilon  = minimum Lperp [cm]  to prevent Lperp -> 0 where Te -> 0
    factor   = proportionality constant  (calibrated against the GBS code)
"""

from math import sqrt

# ---------------------------------------------------------------------------
# Configurable parameters
# ---------------------------------------------------------------------------

# Ion parameters
mu = 2.0    # ion mass / proton mass  (2 = deuterium)
Z  = 1.0    # ion charge state

# Model parameters
epsilon = 0.3    # minimum Lperp [cm]
factor  = 7.4    # Lperp = factor * rho_s  (GBS calibration)

# ---------------------------------------------------------------------------
# Model function
# ---------------------------------------------------------------------------

def scatteringLengthPerp(rho, theta, Ne, Te, Bnorm):
    """Return the perpendicular correlation length Lperp in cm.

    Parameters
    ----------
    rho   : float   normalised poloidal flux
    theta : float   poloidal angle [rad]
    Ne    : float   electron density [m^-3]  (not used, kept for API compatibility)
    Te    : float   electron temperature [keV]
    Bnorm : float   magnetic field strength [T]
    """
    Te_eV   = abs(Te) * 1.e3     # keV -> eV
    Bnorm_G = Bnorm * 1.e4       # T   -> Gauss
    return max(epsilon, factor * 1.02e2 * sqrt(mu * Te_eV / Z) / Bnorm_G)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    rho, theta, Ne, Te, Bnorm = 0.5, 0.5, 1.e19, 1.0, 1.4
    print(f'Lperp = {scatteringLengthPerp(rho, theta, Ne, Te, Bnorm):.4f} cm')
