module qlabs
  ! ---------------------------------------------------------------------------
  ! Fully-relativistic EC absorption with an arbitrary electron distribution
  ! function (EDF) from LUKE Fokker-Planck code.
  !
  ! Physics: Shkarofsky & Shoucri (2011), Comput. Phys. Commun. 182, 1507.
  !   The anti-Hermitian dielectric tensor elements are computed as a single
  !   1D integral along the cyclotron resonance ellipse in (z∥, z⊥) space.
  !
  ! Algorithm:
  !   1.  Call Farina's warmdisp (from ecdisp module) with the bulk Maxwellian
  !       to obtain Re(N⊥) and the polarization eigenvector (ex, ey, ez).
  !       These depend only on the Hermitian (dispersive) part, which is
  !       insensitive to the EDF tail.
  !   2.  Run GL-35 quadrature over the resonance ellipse with the LUKE EDF.
  !   3.  Run the same GL quadrature with the analytical Maxwell-Jüttner EDF
  !       at the same bulk temperature.
  !   4.  absImN = Im(N⊥)_farina × (integral_ql / integral_maxw)
  !       This ratio scales the known-accurate Farina result by the
  !       correction from the non-Maxwellian EDF, cancelling all
  !       normalization constants and coordinate-system factors.
  !
  ! Public interface (mirrors ecdisp/warmdamp.f90):
  !   qlabs_init(p_grid, xi_grid, psi_grid, edf, np, nxi, npsi)
  !     -- load EDF once per run
  !   qlabs_absorption(op, oc, Nr, theta, te, imod, psi_local, b_over_bmin, imNprw)
  !     -- per ray-point absorption coefficient
  !
  ! The output imNprw is Im(N⊥), identical in meaning to warmdamp's output.
  ! ---------------------------------------------------------------------------

  use ecdisp,    only: set_extv, warmdisp, larmornumber
  use qlabs_edf, only: edf_init, edf_eval
  use qlabs_bessel, only: jn_and_jnp

  implicit none
  integer, parameter :: r8 = selected_real_kind(15,300)
  real(r8), parameter :: pi_r8   = 3.1415926535897932384626433832795_r8
  real(r8), parameter :: sqrt_pi = 1.7724538509055160272981674833411_r8

  ! Gauss-Legendre nodes/weights on [-1,1], computed once at first call.
  integer,  parameter :: n_gl = 35
  real(r8), save :: gl_nodes(n_gl), gl_weights(n_gl)
  logical,  save :: gl_init_done = .false.

contains

  ! ===========================================================================
  ! qlabs_init  — load the LUKE EDF; call once before qlabs_absorption
  ! ===========================================================================
  subroutine qlabs_init(p_grid, xi_grid, psi_grid, edf_in, np, nxi, npsi)
    integer,  intent(in) :: np, nxi, npsi
    real(r8), intent(in) :: p_grid(np), xi_grid(nxi), psi_grid(npsi)
    real(r8), intent(in) :: edf_in(np, nxi, npsi)

    call edf_init(p_grid, xi_grid, psi_grid, edf_in, np, nxi, npsi)

    if (.not. gl_init_done) then
      call gauleg(gl_nodes, gl_weights, n_gl)
      gl_init_done = .true.
    end if

  end subroutine qlabs_init


  ! ===========================================================================
  ! qlabs_absorption  — returns Im(N⊥) for the loaded EDF
  !
  !   Inputs:
  !     op          = X = omega_pe^2 / omega^2
  !     oc          = Y^2 = omega_ce^2 / omega^2   (same as warmdamp)
  !     Nr          = |N| = sqrt(N_par^2 + N_perp^2)
  !     theta       = angle between N and B  [rad]
  !     te          = electron temperature   [keV]  (bulk, for warmdisp call)
  !     imod        = wave mode: +1 O-mode, -1 X-mode
  !     psi_local   = normalised poloidal flux at this ray point
  !     b_over_bmin = B/B_min at this poloidal position (for trapping)
  !   Output:
  !     imNprw      = Im(N⊥)  [same sign convention as warmdamp]
  ! ===========================================================================
  subroutine qlabs_absorption(op, oc, Nr, theta, te, imod, psi_local, b_over_bmin, imNprw)

    real(r8), intent(in)  :: op, oc, Nr, theta, te, psi_local, b_over_bmin
    integer,  intent(in)  :: imod
    real(r8), intent(out) :: imNprw

    ! --- local variables ---
    integer  :: nharm, nharm_res, lrm, ierr
    integer, parameter :: fast = 3, imax_harm = 3

    real(r8) :: sqoc, Nll, Nperp_cold, mu_bulk, BTH
    real(r8) :: ZM_vth, ZP_vth, CNST, OMC, PARN, PARN2
    real(r8) :: integral_ql, integral_maxw
    real(r8) :: imNprw_maxw

    complex(r8) :: anpr_warm, ex, ey, ez

    real(r8) :: P11, P22, P33, P12, P13, P32
    ! polarization projection weights (all real; see analysis in plan)
    ! P11 = |ex|², P22 = |ey|², P33 = |ez|²
    ! P12 = Im(ex* ey), P13 = Re(ex* ez), P32 = Im(ey* ez)

    if (te <= 0.0_r8) then
      imNprw = 0.0_r8 ; return
    end if

    ! --- Step A: call warmdisp for polarization and Im(N⊥)_Maxwellian ---------
    call set_extv
    sqoc = sqrt(oc)
    Nll  = Nr * cos(theta)
    Nperp_cold = sqrt(max(0.0_r8, Nr**2 - Nll**2))
    mu_bulk = 511.0_r8 / te
    ! BTH = vth/c = sqrt(kBTe / mec²) = sqrt(Te[keV]/511).
    ! This is the 1D thermal speed normalization used by LUKE and damping.f:
    !   p_th = m_e * vth,  z = p/(m_e vth),  BTH^2 = 1/mu_bulk = kBTe/(m_e c^2).
    ! Note: NOT sqrt(2/mu) — the factor-of-2 convention is not used here.
    BTH = sqrt(1.0_r8 / mu_bulk)

    ! larmornumber returns the TOTAL count of thermally-accessible harmonics;
    ! it is used as lrm for warmdisp (which sums all harmonics 1..lrm).
    nharm = larmornumber(sqoc, Nll, mu_bulk)
    lrm   = min(imax_harm, nharm)

    ! For the Shkarofsky single-harmonic integral, identify the DOMINANT
    ! resonant harmonic nharm_res = nearest integer to omega/omega_ce = 1/sqoc.
    ! This is independent of nharm (which is a count, not a harmonic number).
    ! For ECRH near the fundamental (sqoc ≈ 1), nharm_res = 1.
    ! For second-harmonic heating (sqoc ≈ 0.5), nharm_res = 2.
    nharm_res = max(1, nint(1.0_r8 / sqoc))
    OMC = real(nharm_res, r8) * sqoc   ! n_res * omega_ce/omega ≈ 1

    call warmdisp(op, sqoc, Nll, mu_bulk, -imod, fast, lrm, Nperp_cold, &
                  anpr_warm, ex, ey, ez, ierr)

    if (ierr /= 0) then
      imNprw = 0.0_r8 ; return
    end if

    imNprw_maxw = aimag(anpr_warm)
    if (imNprw_maxw <= 0.0_r8) then
      imNprw = 0.0_r8 ; return
    end if

    ! --- Step B: resonance limits (same as damping.f Eq. 3) ------------------
    PARN  = Nll
    PARN2 = PARN * PARN
    CNST  = PARN2 + OMC**2 - 1.0_r8

    if (CNST <= 0.0_r8) then
      ! Shkarofsky resonance ellipse is degenerate (near-perpendicular propagation
      ! or sub-harmonic resonance).  We cannot compute the EDF-ratio correction
      ! in this regime, but imNprw_maxw (from warmdisp) is still valid.
      ! Physics: at small N‖ the RF-driven tail makes only a ~1 correction; the
      ! Maxwellian result is the best available estimate.
      imNprw = imNprw_maxw ; return
    end if

    ! ZM_vth, ZP_vth are the integration limits in z = p/(mevth) units.
    ! Shkarofsky Eq. (3); valid for |N∥| < 1 (ECRH case).
    ! For |N∥| = 1 exactly: special limits (damping.f lines 274-276):
    !   ZM = (OMC - 1/OMC)/2, ZP = -ZM.
    ! For |N∥| > 1 (EBW): different formula from Eq. (4)-(6), not implemented.
    if (abs(PARN2 - 1.0_r8) < 1.0e-6_r8) then
      ! N∥ ≈ ±1: use the N∥ = 1 limit
      ZM_vth = 0.5_r8 * (OMC - 1.0_r8/OMC) / BTH
      ZP_vth = -ZM_vth
      if (ZM_vth > ZP_vth) then
        ! ensure ZM < ZP
        ZM_vth = -abs(ZM_vth)
        ZP_vth =  abs(ZP_vth)
      end if
    else
      ! General |N∥| ≠ 1 case (Eq. 3)
      ! Denominators: (1-N∥²) = +(1-PARN2) which can be positive or negative.
      ! For |N∥| < 1: 1-N∥² > 0.  For |N∥| > 1: 1-N∥² < 0 (EBW, caveat above).
      ! Using abs() ensures ZM < ZP for |N∥| < 1 (correct formula).
      ZM_vth = (PARN*OMC - sqrt(CNST)) / abs(1.0_r8 - PARN2) / BTH
      ZP_vth = (PARN*OMC + sqrt(CNST)) / abs(1.0_r8 - PARN2) / BTH
    end if
    ! Both in z_vth units (p_par / m_e v_th)

    if (ZP_vth <= ZM_vth) then
      imNprw = 0.0_r8 ; return
    end if

    ! --- Step C: polarization projection weights -----------------------------
    P11 = real(ex,r8)**2 + aimag(ex)**2     ! |ex|²
    P22 = real(ey,r8)**2 + aimag(ey)**2     ! |ey|²
    P33 = real(ez,r8)**2 + aimag(ez)**2     ! |ez|²
    ! Im(ex* × ey) = Im(conj(ex)*ey)
    P12 = real(ex,r8)*aimag(ey) - aimag(ex)*real(ey,r8)
    ! Re(ex* × ez) = Re(conj(ex)*ez)
    P13 = real(ex,r8)*real(ez,r8) + aimag(ex)*aimag(ez)
    ! Im(ey* × ez) = Im(conj(ey)*ez)
    P32 = real(ey,r8)*aimag(ez) - aimag(ey)*real(ez,r8)

    ! --- Step D: GL quadrature over [ZM_vth, ZP_vth] -------------------------
    call resonance_integral(OMC, PARN, BTH, mu_bulk, &
                            real(anpr_warm, r8),       &    ! Re(N_perp) from warmdisp
                            nharm_res, ZM_vth, ZP_vth, &
                            psi_local, b_over_bmin,    &
                            P11, P22, P33, P12, P13, P32, &
                            integral_ql, integral_maxw)

    ! --- Step E: scale Farina result by EDF ratio -----------------------------
    if (abs(integral_maxw) < 1.0e-30_r8) then
      imNprw = 0.0_r8 ; return
    end if

    imNprw = imNprw_maxw * integral_ql / integral_maxw

    ! do not allow negative values (no net emission in this module)
    if (imNprw < 0.0_r8) imNprw = 0.0_r8

  end subroutine qlabs_absorption


  ! ===========================================================================
  ! resonance_integral  — GL quadrature on [ZM_vth, ZP_vth]
  !
  !   Returns the polarization-projected resonance integrals for:
  !     integral_ql    : LUKE EDF
  !     integral_maxw  : analytical Maxwell-Juettner at temperature 1/mu_bulk
  !
  !   Both integrals have identical Bessel/polarization kernels; only the
  !   DERIVF (QL gradient of f) differs. The ratio I_ql/I_maxw cancels all
  !   normalization constants.
  !
  !   K_proj = V11*P11 + V22*P22 + V33*P33 + 2*V12*P12 + 2*V13*P13 + 2*V32*P32
  !   integrand = ZPERP * K_proj * DERIVF
  ! ===========================================================================
  subroutine resonance_integral(OMC, PARN, BTH, mu_bulk, Nperp_real, nharm, &
                                 ZM_vth, ZP_vth, psi_local, b_over_bmin,     &
                                 P11, P22, P33, P12, P13, P32,               &
                                 integral_ql, integral_maxw)

    real(r8), intent(in)  :: OMC, PARN, BTH, mu_bulk, Nperp_real
    integer,  intent(in)  :: nharm
    real(r8), intent(in)  :: ZM_vth, ZP_vth, psi_local, b_over_bmin
    real(r8), intent(in)  :: P11, P22, P33, P12, P13, P32
    real(r8), intent(out) :: integral_ql, integral_maxw

    integer  :: k
    real(r8) :: mid, half_range, z_vth, ZPAR, ZARG, ZPERP
    real(r8) :: p_sph, xi_sph, xi_local_sq, xi_bmin_sq
    real(r8) :: u_total, gamma_res
    real(r8) :: Xarg, Jn, Jnp
    real(r8) :: V11, V12, V13, V22, V32, V33, K_proj
    real(r8) :: f_val, dfdu_par, dfdu_perp, DERIVF_ql
    real(r8) :: f_mj, DERIVF_mj
    real(r8) :: n_r  ! harmonic number as real

    integral_ql   = 0.0_r8
    integral_maxw = 0.0_r8

    mid        = 0.5_r8 * (ZM_vth + ZP_vth)
    half_range = 0.5_r8 * (ZP_vth - ZM_vth)
    n_r        = real(nharm, r8)

    do k = 1, n_gl
      ! GL node in [ZM_vth, ZP_vth]
      z_vth = mid + half_range * gl_nodes(k)   ! z_par in v_th units
      ZPAR  = z_vth * BTH                      ! u_par = p_par/(m_e c)

      ! Resonance curve: ZPERP = u_perp_res from Shkarofsky Eq. (2)
      ZARG = (PARN**2 - 1.0_r8)*ZPAR**2 + 2.0_r8*PARN*OMC*ZPAR + OMC**2 - 1.0_r8
      if (ZARG <= 0.0_r8) cycle

      ZPERP = sqrt(ZARG)     ! u_perp_res in m_e c units

      ! Convert to spherical (p, xi) for EDF lookup
      u_total = sqrt(ZPAR**2 + ZPERP**2)
      p_sph   = u_total                 ! u = p/(m_e c) — grid stored in p_mc units

      ! xi_local = local pitch-angle cosine at the observation field B_local
      ! LUKE stores f(p, xi_Bmin, psi) where xi is defined at B_min.
      ! Mirror mapping: xi_local^2 = 1 - (B/Bmin)*(1 - xi_Bmin^2)
      !               → xi_Bmin^2  = 1 - (1/B_over_Bmin)*(1 - xi_local^2)
      ! If xi_Bmin^2 < 0: electron is magnetically trapped → skip this node.
      !
      ! Special case B_over_Bmin = 1 (observation at Bmin): xi_Bmin = xi_local.
      xi_sph = ZPAR / u_total     ! xi_local
      if (b_over_bmin > 1.0_r8 + 1.0e-10_r8) then
        ! Mirror mapping: compute xi at Bmin from xi at B_local
        ! z_par = p_par/(m_e c), ZPERP = p_perp/(m_e c), so xi_local = z_par/u
        ! xi_Bmin² = 1 - (1/R_mirror) × (1 - xi_local²)
        ! where R_mirror = B_local/Bmin = b_over_bmin
        xi_local_sq = xi_sph**2
        xi_bmin_sq  = 1.0_r8 - (1.0_r8 / b_over_bmin) * (1.0_r8 - xi_local_sq)
        if (xi_bmin_sq < 0.0_r8) cycle   ! trapped electron: skip
        xi_sph = sign(1.0_r8, xi_sph) * sqrt(xi_bmin_sq)  ! xi_Bmin
      end if

      gamma_res = sqrt(1.0_r8 + u_total**2)

      ! --- Bessel functions and polarization kernel --------------------------
      Xarg = Nperp_real * ZPERP * n_r / OMC   ! Shkarofsky Eq. (8) argument

      call jn_and_jnp(nharm, Xarg, Jn, Jnp)

      ! V_ij basis terms (following damping.f sign convention).
      ! For small Xarg, use the analytic small-argument expansion to avoid 0/0:
      !   J_n(x) ~ (x/2)^n / n!,  J_n'(x) ~ n*(x/2)^(n-1)/(2*n!)
      !   n*J_n/x ~ (x/2)^(n-1) / (2*(n-1)!)
      ! For n=1: n*J_1/x → 1/2,  J_1' → 1/2  (both = 1/2 as x→0).
      ! For n>1: all V_ij → 0 (higher-order in x).
      ! jn_and_jnp already handles x < 1e-10 correctly, returning Jn = x/2 (n=1).
      ! We only need the special branch for absolute zero (Xarg = 0 exactly).
      if (Xarg < 1.0e-10_r8) then
        if (nharm == 1) then
          ! n=1 small-argument limits: n*J_1/x → 1/2, J_1' → 1/2
          V11 = 0.25_r8   ! (1/2)^2
          V22 = 0.25_r8   ! (1/2)^2
          V12 = 0.25_r8   ! 1 * (x/2)/x * (1/2) → 1/4
        else
          ! n >= 2: all V_ij → 0 as x → 0; integrand → 0 as ZPERP → 0
          cycle
        end if
        V13 = 0.0_r8 ; V33 = 0.0_r8 ; V32 = 0.0_r8
      else
        V11 = (n_r * Jn / Xarg)**2
        V13 = (ZPAR / ZPERP) * n_r * Jn**2 / Xarg
        V33 = (ZPAR / ZPERP * Jn)**2
        V12 = n_r * Jn * Jnp / Xarg
        V22 = Jnp**2
        V32 = -(ZPAR / ZPERP) * Jn * Jnp
      end if

      ! Polarization-projected kernel K_proj
      ! (using the sign structure derived from damping.f normalization)
      K_proj = V11*P11 + V22*P22 + V33*P33 &
             + 2.0_r8*V12*P12               &   ! off-diagonal 1-2
             + 2.0_r8*V13*P13               &   ! off-diagonal 1-3
             + 2.0_r8*V32*P32               !   off-diagonal 3-2

      if (K_proj == 0.0_r8) cycle

      ! --- LUKE EDF: evaluate f and derivatives at the resonance point -------
      ! p_sph = u_total = p/(m_e c); edf_eval returns df/du_par, df/du_perp.
      call edf_eval(p_sph, xi_sph, psi_local, b_over_bmin, &
                    f_val, dfdu_par, dfdu_perp)

      ! QL gradient operator in u = p/(m_e c) coordinates
      ! (ratio-based approach: BTH factor cancels between I_ql and I_maxw)
      DERIVF_ql = dfdu_perp * OMC + dfdu_par * ZPERP * PARN

      ! --- Maxwell-Juettner at the same (u_par, u_perp): analytical ----------
      ! f is expected to be normalized to 1 (Shkarofsky convention).
      ! The analytical MJ normalized to 1 is exp(-mu*(gamma-1)) / (4π Θ K₂(1/Θ)),
      ! but the normalization constant cancels in the ratio I_ql/I_maxw,
      ! so we only need the exponential part.
      f_mj = exp(-mu_bulk * (gamma_res - 1.0_r8))
      DERIVF_mj = -f_mj * mu_bulk / gamma_res * &
                  (ZPERP * OMC + ZPAR * ZPERP * PARN)

      ! --- Accumulate (GL weight × half_range converts [-1,1] → [ZM,ZP]) ---
      integral_ql   = integral_ql   + half_range * gl_weights(k) * ZPERP * K_proj * DERIVF_ql
      integral_maxw = integral_maxw + half_range * gl_weights(k) * ZPERP * K_proj * DERIVF_mj

    end do

  end subroutine resonance_integral


  ! ===========================================================================
  ! gauleg  — Gauss-Legendre nodes and weights on [-1, 1]
  !   Newton-Raphson on Legendre polynomial roots.
  !   Classic algorithm from Numerical Recipes in Fortran (Press et al.).
  ! ===========================================================================
  subroutine gauleg(x, w, n)
    integer,  intent(in)  :: n
    real(r8), intent(out) :: x(n), w(n)

    real(r8), parameter :: eps = 3.0e-14_r8
    integer  :: i, j, m
    real(r8) :: p1, p2, p3, pp, xl, xm, z, z1

    m  = (n + 1) / 2
    xm = 0.0_r8
    xl = 1.0_r8

    do i = 1, m
      z = cos(pi_r8 * (i - 0.25_r8) / (n + 0.5_r8))   ! initial guess
      z1 = 2.0_r8 * z

      do while (abs(z - z1) > eps)
        p1 = 1.0_r8
        p2 = 0.0_r8
        do j = 1, n
          p3 = p2
          p2 = p1
          p1 = ((2*j - 1) * z * p2 - (j - 1) * p3) / j
        end do
        ! p1 = P_n(z),  pp = P_n'(z)
        pp = n * (z * p1 - p2) / (z * z - 1.0_r8)
        z1 = z
        z  = z1 - p1 / pp
      end do

      x(i)     = xm - xl * z
      x(n+1-i) = xm + xl * z
      w(i)     = 2.0_r8 * xl / ((1.0_r8 - z*z) * pp*pp)
      w(n+1-i) = w(i)
    end do

  end subroutine gauleg

end module qlabs
