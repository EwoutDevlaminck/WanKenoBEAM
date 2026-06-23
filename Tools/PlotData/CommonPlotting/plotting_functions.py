
"""
Collection of commonly used plotting functions.
"""

import warnings
import numpy as np


# We want to plot the contours corresponding to the
# the cyclotrob harmonics omega = n omega_ce for n=1,2,3, the cut-offs
# and other wave-wave resonances. Depending on the frequency the some or
# all contours might be empty returning in a warning.
# We want to catch the warning and issue better information to the user ...

# ... All warnings triggered ...
warnings.simplefilter("always")

# ... The warning message to catch is...
wmsg = 'No contour levels were found within the data range.'


# Plotting the cyclotron resonances
def add_cyclotron_resonances(R, Z, StixY, axes):

    """
    Plot the locus of point that satisfy the resonance condition StixY = 1/n,
    for n=1,2,3, where StixY = omega_ce/omega.

    Usage: 
           h1, h2, h3 = add_cyclotron_resonances(R, Z, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane and StixY is the
    Stix parameter X = omega_ce/omega as a function of (R,Z). The last argument
    is the axes to be polulated.  

    The returned objects h1, h2, h3 refer to the first, second and third harmonics
    respectively and can be used, for instance, to build a colorbar or a legend.
    """

    with warnings.catch_warnings(record=True) as warn:
        
        # This counter is used to check if new warnings are issued
        len_warn = 0

        # Try to plot the third-harmonic resonance ...
        h3 = axes.contour(R, Z, StixY, [0.33], colors='lime', linestyles='dashed')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:
            
            print('The third-haronic resonance is not in the domain.')
            len_warn += 1

        # ... try to plot the second-harmonic resonance ...
        h2 = axes.contour(R, Z, StixY, [0.50], colors='lime', linestyles='dashdot')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:

            print('The second-haronic resonance is not in the domain.')
            len_warn += 1

        # ... try to plot the first-harmonic resonance ...
        h1 = axes.contour(R, Z, StixY, [1.00], colors='lime', linestyles='dotted')
        # ... check the last (-1) warning ...
        if len(warn) > len_warn and str(warn[0].message) == wmsg:

            print('The first-haronic resonance is not in the domain.')
    
    return h1, h2, h3


# Plotting the O-mode cutoff    
def add_Omode_cutoff(R, Z, StixX, axes):

    """
    Add to a plot in the R,Z plane, a curve for the level set StixX = 1,
    which corresponds to the O-mode cutoff.

    Usage: 
            contour = add_Omode_cutoff(R, Z, StixX, axes)
           
    where R, Z are coordinates in the poloidal plane and StixX is the
    Stix parameter Y = omega_pe^2/omega^2 as a function of (R,Z). The last argument
    is the axes to be polulated.  

    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    # If present in the domain add line for the O-mode cut-off which is 
    # the level set StixX == 1.

    # If the O-mode is not in the domain, the level set is empty and
    # a warning is issued. We want to catch the warning
    with warnings.catch_warnings(record=True) as warn:

        # ... O-mode cut-off
        O_cutoff = axes.contour(R, Z, StixX, [1.], colors='g', linestyles='dashed')

        # ... check the last (-1) warning ...
        if len(warn)>0 and str(warn[-1].message) == wmsg:
            print('The O-Mode cut-off surface not found in the domain.')
            
    return O_cutoff


def add_Xmode_cutoff(R, Z, StixX, StixY, axes):

    """
    Add a plot of the X-mode cutoff to the give axes. The X-mode cutoff is given
    by the condition
    
       (Y/2) + sqrt(X + (Y/2)^2) = 1,
    
    where X and Y are the standard Stix parameters.

    Usage: 
           contour = add_Xmode_cutoff(R, Z, StixX, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane, whereas StixX and StixY 
    are the Stix parameters X = omega_pe^2/omega^2 and Y = omega_ce/omega 
    as functions of (R,Z). The last argument is the axes to be polulated. 
    
    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    Xcutoff = 0.5*StixY + np.sqrt(StixX + (0.5*StixY)**2)

    # If the level set is empty, the UH resonance is not present in the domain
    # and a warning is issued. We want to catch this warning
    with warnings.catch_warnings(record=True) as warn:

        # ... Upper-hybrid resonance for perpendicular propagation
        X_cutoff = axes.contour(R, Z, Xcutoff, [1.0], colors='g')

        # ... check the last (-1) warning ...
        if len(warn) > 0 and str(warn[-1].message) == wmsg:
            print('The O-Mode cut-off surface not found in the domain.')
            
    return X_cutoff


# Computing the characteristic radii for a top-down (X,Y) overlay
def topdown_radii(Eq, FreqGHz, harm_n=2, ntheta=200):

    """
    Compute the characteristic major radii needed to overlay a top-down
    (X,Y) plot with circles: the magnetic axis, the inboard/outboard extent
    of the LCFS (psi=1), and the cold n-th harmonic resonance radius on the
    midplane (StixY = omega_ce/omega = 1/n).

    Usage:
           R_axis, R_lcfs_min, R_lcfs_max, R_harm = \\
               topdown_radii(Eq, FreqGHz, harm_n, ntheta)

    where Eq is an equilibrium object (e.g. TokamakEquilibrium) and FreqGHz
    is the wave frequency in GHz. R_harm is None if the resonance condition
    is not satisfied anywhere on the midplane within the LCFS R-range.
    """

    import scipy.optimize as opt
    from CommonModules.PlasmaEquilibrium import StixParamSample

    R_axis, Z_axis = Eq.magn_axis_coord_Rz

    # LCFS (psi=1) extent: sample (psi=1, theta) -> (R, Z) over the full
    # poloidal angle and take the min/max R reached.
    theta = np.linspace(0., 2.*np.pi, ntheta, endpoint=False)
    R_lcfs = np.array([Eq.flux_to_grid_coord(1.0, th)[0] for th in theta])
    R_lcfs_min, R_lcfs_max = R_lcfs.min(), R_lcfs.max()

    # Cold n-th harmonic resonance on the midplane: StixY(R, Z_axis) = 1/n.
    # Sample StixY along R at fixed Z = Z_axis and bracket the first sign
    # change of (StixY - 1/n), then refine with a root finder.
    R1d = np.linspace(R_lcfs_min, R_lcfs_max, 400)
    _, StixY1d, _ = StixParamSample(R1d, np.array([Z_axis]), Eq, FreqGHz)
    target = 1.0 / harm_n
    f = StixY1d[0, :] - target

    R_harm = None
    sign_change = np.where(np.diff(np.sign(f)) != 0)[0]
    if len(sign_change) > 0:
        i = sign_change[0]

        def _f(R):
            _, sY, _ = StixParamSample(np.array([R]), np.array([Z_axis]), Eq, FreqGHz)
            return sY[0, 0] - target

        R_harm = opt.brentq(_f, R1d[i], R1d[i+1])

    return R_axis, R_lcfs_min, R_lcfs_max, R_harm


# Plotting the top-down (X,Y) overlay circles
def add_topdown_circles(axes, R_axis, R_lcfs_min, R_lcfs_max, R_harm=None, harm_n=2):

    """
    Overlay the magnetic axis, LCFS inboard/outboard extent, and (optionally)
    a cold harmonic resonance radius as circles centred on the device axis
    (X=Y=0), on a top-down (X,Y) plot.

    Usage:
           h_axis, h_lcfs, h_harm = \\
               add_topdown_circles(axes, R_axis, R_lcfs_min, R_lcfs_max, R_harm, harm_n)

    R_axis, R_lcfs_min, R_lcfs_max, R_harm are major radii in the same units
    as the (X,Y) axes (cm, typically), as returned by topdown_radii(). Pass
    R_harm=None to skip the resonance circle. harm_n only sets the line
    style/label (matching add_cyclotron_resonances: 1->dotted, 2->dashdot,
    3->dashed). axes is the matplotlib Axes to draw on.

    Returns (h_axis, (h_lcfs_in, h_lcfs_out), h_harm); h_harm is None if
    R_harm was None.
    """

    theta = np.linspace(0., 2.*np.pi, 200)
    cst, snt = np.cos(theta), np.sin(theta)

    h_axis, = axes.plot(R_axis*cst, R_axis*snt,
                         color='grey', linestyle='dashed', linewidth=1,
                         label='Magnetic axis')
    h_lcfs_in,  = axes.plot(R_lcfs_min*cst, R_lcfs_min*snt,
                             color='r', linewidth=1, label='LCFS')
    h_lcfs_out, = axes.plot(R_lcfs_max*cst, R_lcfs_max*snt,
                             color='r', linewidth=1)

    h_harm = None
    if R_harm is not None:
        style = {1: 'dotted', 2: 'dashdot', 3: 'dashed'}.get(harm_n, 'dashdot')
        h_harm, = axes.plot(R_harm*cst, R_harm*snt, color='lime', linestyle=style,
                             label='{}-th harmonic'.format(harm_n))

    return h_axis, (h_lcfs_in, h_lcfs_out), h_harm


def add_UHresonance(R, Z, StixX, StixY, axes):

    """
    Plot the upper-hybrid resonance for the case of exactly perpendicular
    propagation as an indication on the position of the upper-hybrid resonance.

    Usage: 
            contour = add_UHresonance(R, Z, StixX, StixY, axes)
           
    where R, Z are coordinates in the poloidal plane, whereas StixX and StixY 
    are the Stix parameters X = omega_pe^2/omega^2 and Y = omega_ce/omega 
    as functions of (R,Z). The last argument is the axes to be polulated.  

    The returned object can be used, for instance, to build a colorbar or a legend.
    """

    # For perpendicular propagation, the upper-hybrid (UH) resonance is
    # the level set StixX + StixY**2 == 1. If present in the domain, add
    # this contour line to the axes. 

    # If the level set is empty, the UH resonance is not present in the domain
    # and a warning is issued. We want to catch this warning
    with warnings.catch_warnings(record=True) as warn:

        # ... Upper-hybrid resonance for perpendicular propagation
        UH_res = axes.contour(R, Z, StixX + StixY**2, [1.0], colors='m')

        # ... check the last (-1) warning ...
        if len(warn)>0 and str(warn[-1].message) == wmsg:
            print('The upper-hybrid resonance X+Y^2=1 is not in the domain.')
    
    return UH_res
