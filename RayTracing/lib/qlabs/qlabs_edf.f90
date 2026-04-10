module qlabs_edf
  ! ---------------------------------------------------------------------------
  ! EDF storage and evaluation for the qlabs absorption module.
  !
  ! The EDF from LUKE is stored on a spherical momentum grid (p, xi, psi):
  !   p   -- total normalised momentum  p/(m_e c),  p >= 0   [relativistic units]
  !   xi  -- pitch-angle cosine at Bmin,  xi in [-1, 1]
  !   psi -- normalised poloidal flux,    psi in [0, 1]
  !
  ! Using p/(m_e c) (rather than the thermal unit p/(m_e v_th)) means the grid
  ! is fixed and independent of local electron temperature.  The caller is
  ! responsible for converting the LUKE thermal grid before calling edf_init:
  !   p_mc(i) = p_th(i) * beta,   beta = v_th/c = sqrt(Te_ref[keV] / 511).
  !
  ! Internally we store  log(f)  for numerical stability (same convention as
  ! YodaU/Modules/edf_interpolation.py).  Values where f <= 0 are replaced by
  ! log_floor = -500 so that exp(log_floor) = 0 to machine precision.
  !
  ! Derivatives:
  !   d(log f)/dp  and  d(log f)/dxi  are computed by natural cubic splines
  !   (Thomas algorithm) along each 1-D pencil independently, then saved in
  !   module arrays.  The chain rule converts them to (u_par, u_perp) at
  !   call time, where u = p/(m_e c).
  !
  ! Trapping:
  !   At a poloidal position where B/Bmin > 1, trapped electrons with
  !   |xi_Bmin| < xi_trap = sqrt(1 - 1/B_over_Bmin) do not contribute to
  !   local absorption.  The caller passes B_over_Bmin; the evaluation
  !   subroutine returns f = 0 (and zero derivatives) for trapped points.
  ! ---------------------------------------------------------------------------

  implicit none
  integer, parameter :: r8 = selected_real_kind(15,300)

  ! --- Module-level saved arrays (allocated once by edf_init) ----------------
  integer,  save :: np_g, nxi_g, npsi_g
  real(r8), allocatable, save :: p_g(:)     ! (np_g)
  real(r8), allocatable, save :: xi_g(:)    ! (nxi_g)
  real(r8), allocatable, save :: psi_g(:)   ! (npsi_g)
  real(r8), allocatable, save :: logf_g(:,:,:)     ! (np_g, nxi_g, npsi_g)
  real(r8), allocatable, save :: dlogf_dp_g(:,:,:)  ! spline d/dp
  real(r8), allocatable, save :: dlogf_dxi_g(:,:,:) ! spline d/dxi

  real(r8), parameter :: log_floor = -500.0_r8

contains

  ! ===========================================================================
  ! edf_init  — called once from qlabs_init to load and preprocess the EDF
  ! ===========================================================================
  subroutine edf_init(p_in, xi_in, psi_in, f_in, np, nxi, npsi)
    integer,  intent(in) :: np, nxi, npsi
    real(r8), intent(in) :: p_in(np), xi_in(nxi), psi_in(npsi)
    real(r8), intent(in) :: f_in(np, nxi, npsi)

    integer  :: ip, ixi, ipsi
    real(r8) :: fval

    ! --- store grid ---
    np_g   = np
    nxi_g  = nxi
    npsi_g = npsi

    if (allocated(p_g))             deallocate(p_g, xi_g, psi_g)
    if (allocated(logf_g))          deallocate(logf_g)
    if (allocated(dlogf_dp_g))      deallocate(dlogf_dp_g)
    if (allocated(dlogf_dxi_g))     deallocate(dlogf_dxi_g)

    allocate(p_g(np), xi_g(nxi), psi_g(npsi))
    allocate(logf_g(np, nxi, npsi))
    allocate(dlogf_dp_g(np, nxi, npsi))
    allocate(dlogf_dxi_g(np, nxi, npsi))

    p_g   = p_in
    xi_g  = xi_in
    psi_g = psi_in

    ! --- compute log(f) with floor ---
    do ipsi = 1, npsi
      do ixi = 1, nxi
        do ip = 1, np
          fval = f_in(ip, ixi, ipsi)
          if (fval > 0.0_r8) then
            logf_g(ip, ixi, ipsi) = log(fval)
          else
            logf_g(ip, ixi, ipsi) = log_floor
          end if
        end do
      end do
    end do

    ! --- precompute spline derivatives d(log f)/dp along p-pencils ---
    do ipsi = 1, npsi
      do ixi = 1, nxi
        call natural_spline_deriv(p_g, logf_g(:, ixi, ipsi), np, &
                                   dlogf_dp_g(:, ixi, ipsi))
      end do
    end do

    ! --- precompute spline derivatives d(log f)/dxi along xi-pencils ---
    do ipsi = 1, npsi
      do ip = 1, np
        call natural_spline_deriv(xi_g, logf_g(ip, :, ipsi), nxi, &
                                   dlogf_dxi_g(ip, :, ipsi))
      end do
    end do

  end subroutine edf_init


  ! ===========================================================================
  ! edf_eval  — evaluate f and cylindrical derivatives at a single point
  !
  !   Inputs (spherical):
  !     p_s         = u = p/(m_e c)  [relativistic momentum modulus]
  !     xi_s        = pitch-angle cosine at Bmin
  !     psi_s       = normalised poloidal flux
  !     B_over_Bmin = B_local / B_min (trapping ratio)
  !   Outputs:
  !     f_val    -- f (not log f)
  !     dfdu_par  -- df/du_par  where u_par = p_par/(m_e c)
  !     dfdu_perp -- df/du_perp where u_perp = p_perp/(m_e c)
  ! ===========================================================================
  subroutine edf_eval(p_s, xi_s, psi_s, B_over_Bmin, f_val, dfdu_par, dfdu_perp)
    real(r8), intent(in)  :: p_s, xi_s, psi_s, B_over_Bmin
    real(r8), intent(out) :: f_val, dfdu_par, dfdu_perp

    real(r8) :: logf, dlogf_du, dlogf_dxi, xi_trap
    real(r8) :: sin_theta

    ! --- out-of-bounds check for p_s -----------------------------------------
    ! p_s = u = p/(m_e c).  If it exceeds the stored grid the EDF is at or
    ! below its stored floor (the LUKE grid must cover all dynamically relevant
    ! momenta).  Return zero so that this GL node contributes nothing to both
    ! I_QL and I_Maxw.  Without this guard, bisect() clamps to the last cell
    ! and trilinear_eval returns the grid-boundary f value, while DERIVF_mj at
    ! large u is Boltzmann-suppressed to ~0, making the ratio I_QL/I_Maxw diverge.
    if (p_s > p_g(np_g)) then
      f_val = 0.0_r8 ; dfdu_par = 0.0_r8 ; dfdu_perp = 0.0_r8
      return
    end if

    ! --- trapping check -------------------------------------------------------
    ! Trapped electrons have |xi_Bmin| < xi_trap; they don't contribute here.
    ! B_over_Bmin = 1 means no trapping (midplane or exactly at Bmin).
    if (B_over_Bmin > 1.0_r8 + 1.0e-10_r8) then
      xi_trap = sqrt(max(0.0_r8, 1.0_r8 - 1.0_r8 / B_over_Bmin))
      if (abs(xi_s) <= xi_trap) then
        f_val = 0.0_r8 ; dfdu_par = 0.0_r8 ; dfdu_perp = 0.0_r8
        return
      end if
    end if

    ! --- trilinear interpolation of logf and derivatives ----------------------
    call trilinear_eval(p_s, xi_s, psi_s, &
                        logf_g,     logf,     &
                        dlogf_dp_g, dlogf_du, &
                        dlogf_dxi_g, dlogf_dxi)

    f_val = exp(logf)

    ! --- chain rule: (u, xi) -> (u_par, u_perp), where u = p/(m_e c) ----------
    ! u_par  = u * xi          =>  du_par / du = xi,   du_par / dxi = u
    ! u_perp = u * sqrt(1-xi²) =>  du_perp/du = sqrt(1-xi²),
    !                               du_perp/dxi = -u*xi/sqrt(1-xi²)
    !
    ! df/du_par  = df/du * xi + df/dxi * (1-xi²)/u
    ! df/du_perp = df/du * sqrt(1-xi²) - df/dxi * xi*sqrt(1-xi²) / u
    !
    ! Using log derivatives:  df/du = f * d(logf)/du,  etc.

    sin_theta = sqrt(max(0.0_r8, 1.0_r8 - xi_s**2))

    dfdu_par  = f_val * (dlogf_du * xi_s &
                         + dlogf_dxi * (1.0_r8 - xi_s**2) / max(p_s, 1.0e-30_r8))
    dfdu_perp = f_val * sin_theta * (dlogf_du &
                         - dlogf_dxi * xi_s / max(p_s, 1.0e-30_r8))

  end subroutine edf_eval


  ! ===========================================================================
  ! trilinear_eval  — interpolate logf and its pre-computed spline derivatives
  !                   at an arbitrary (p, xi, psi) point.
  ! ===========================================================================
  subroutine trilinear_eval(p_s, xi_s, psi_s, &
                             lf, lf_val, dlf_dp_arr, dlf_dp, dlf_dxi_arr, dlf_dxi)
    real(r8), intent(in)  :: p_s, xi_s, psi_s
    real(r8), intent(in)  :: lf(np_g, nxi_g, npsi_g)
    real(r8), intent(in)  :: dlf_dp_arr(np_g, nxi_g, npsi_g)
    real(r8), intent(in)  :: dlf_dxi_arr(np_g, nxi_g, npsi_g)
    real(r8), intent(out) :: lf_val, dlf_dp, dlf_dxi

    integer  :: ip, ixi, ipsi
    real(r8) :: tp, txi, tpsi

    ! binary search in each dimension (returns lower-bound index)
    ip   = bisect(p_g,   np_g,   p_s)
    ixi  = bisect(xi_g,  nxi_g,  xi_s)
    ipsi = bisect(psi_g, npsi_g, psi_s)

    ! fractional positions within the cell
    tp   = (p_s   - p_g(ip))   / max(p_g(ip+1)   - p_g(ip),   1.0e-30_r8)
    txi  = (xi_s  - xi_g(ixi)) / max(xi_g(ixi+1) - xi_g(ixi), 1.0e-30_r8)
    tpsi = (psi_s - psi_g(ipsi)) / max(psi_g(ipsi+1) - psi_g(ipsi), 1.0e-30_r8)

    ! clamp to [0,1]
    tp   = max(0.0_r8, min(1.0_r8, tp))
    txi  = max(0.0_r8, min(1.0_r8, txi))
    tpsi = max(0.0_r8, min(1.0_r8, tpsi))

    lf_val  = trilin(lf,        ip, ixi, ipsi, tp, txi, tpsi)
    dlf_dp  = trilin(dlf_dp_arr, ip, ixi, ipsi, tp, txi, tpsi)
    dlf_dxi = trilin(dlf_dxi_arr, ip, ixi, ipsi, tp, txi, tpsi)

  end subroutine trilinear_eval


  ! ===========================================================================
  ! trilin  — 8-corner trilinear interpolation (pure function for one array)
  ! ===========================================================================
  real(r8) function trilin(arr, i0, j0, k0, tp, tq, tr)
    real(r8), intent(in) :: arr(np_g, nxi_g, npsi_g)
    integer,  intent(in) :: i0, j0, k0
    real(r8), intent(in) :: tp, tq, tr   ! fractional positions in [0,1]

    integer  :: i1, j1, k1
    real(r8) :: c000, c100, c010, c110, c001, c101, c011, c111

    i1 = min(i0+1, np_g)
    j1 = min(j0+1, nxi_g)
    k1 = min(k0+1, npsi_g)

    c000 = arr(i0, j0, k0)
    c100 = arr(i1, j0, k0)
    c010 = arr(i0, j1, k0)
    c110 = arr(i1, j1, k0)
    c001 = arr(i0, j0, k1)
    c101 = arr(i1, j0, k1)
    c011 = arr(i0, j1, k1)
    c111 = arr(i1, j1, k1)

    trilin = (1.0_r8-tp)*((1.0_r8-tq)*((1.0_r8-tr)*c000 + tr*c001) &
                          +         tq *((1.0_r8-tr)*c010 + tr*c011)) &
           +          tp *((1.0_r8-tq)*((1.0_r8-tr)*c100 + tr*c101) &
                          +         tq *((1.0_r8-tr)*c110 + tr*c111))

  end function trilin


  ! ===========================================================================
  ! bisect  — return lower-bound index i such that  grid(i) <= val < grid(i+1)
  !           Clamps to [1, n-1].
  ! ===========================================================================
  integer function bisect(grid, n, val)
    integer,  intent(in) :: n
    real(r8), intent(in) :: grid(n), val

    integer :: lo, hi, mid

    if (val <= grid(1))   then; bisect = 1;     return; end if
    if (val >= grid(n))   then; bisect = n - 1; return; end if

    lo = 1 ; hi = n
    do while (hi - lo > 1)
      mid = (lo + hi) / 2
      if (val >= grid(mid)) then
        lo = mid
      else
        hi = mid
      end if
    end do
    bisect = lo

  end function bisect


  ! ===========================================================================
  ! natural_spline_deriv — compute first derivatives of y(x) using a natural
  !   cubic spline (zero second derivative at endpoints).  Thomas algorithm.
  !   On exit  dy(i) = dy/dx at x(i).
  ! ===========================================================================
  subroutine natural_spline_deriv(x, y, n, dy)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: x(n), y(n)
    real(r8), intent(out) :: dy(n)

    real(r8) :: h(n-1), alpha(n), l(n), mu_v(n), z(n)
    integer  :: i

    if (n < 3) then
      ! fallback: central/forward/backward differences
      if (n == 2) then
        dy(1) = (y(2) - y(1)) / max(x(2) - x(1), 1.0e-30_r8)
        dy(2) = dy(1)
      else
        dy(1) = 0.0_r8
      end if
      return
    end if

    ! step sizes
    do i = 1, n-1
      h(i) = x(i+1) - x(i)
    end do

    ! right-hand side
    do i = 2, n-1
      alpha(i) = 3.0_r8 * ((y(i+1) - y(i)) / h(i) - (y(i) - y(i-1)) / h(i-1))
    end do

    ! Thomas algorithm for the second-derivative moments M(i)
    l(1)   = 1.0_r8 ; mu_v(1) = 0.0_r8 ; z(1) = 0.0_r8
    do i = 2, n-1
      l(i)   = 2.0_r8 * (x(i+1) - x(i-1)) - h(i-1) * mu_v(i-1)
      mu_v(i) = h(i) / l(i)
      z(i)   = (alpha(i) - h(i-1) * z(i-1)) / l(i)
    end do
    l(n) = 1.0_r8 ; z(n) = 0.0_r8

    ! back-substitution: z(i) becomes the second derivative M(i)
    ! (here we reuse z as M)
    do i = n-1, 1, -1
      z(i) = z(i) - mu_v(i) * z(i+1)
    end do

    ! convert second-derivative moments to first derivatives
    ! y'(i) = (y(i+1)-y(i))/h(i) - h(i)*(2*M(i)+M(i+1))/3   for i<n
    ! y'(n) = (y(n)-y(n-1))/h(n-1) + h(n-1)*(2*M(n)+M(n-1))/3
    do i = 1, n-1
      dy(i) = (y(i+1) - y(i)) / h(i) - h(i) * (2.0_r8*z(i) + z(i+1)) / 3.0_r8
    end do
    dy(n) = (y(n) - y(n-1)) / h(n-1) + h(n-1) * (2.0_r8*z(n) + z(n-1)) / 3.0_r8

  end subroutine natural_spline_deriv

end module qlabs_edf
