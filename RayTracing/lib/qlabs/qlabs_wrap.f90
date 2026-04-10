! qlabs_wrap.f90
! ===========================================================================
! Thin wrappers that expose module qlabs procedures as plain Fortran subroutines
! with simple Fortran name-mangling (qlabs_init_, qlabs_absorption_) so that
! f2py can link against them directly.
!
! Background
! ----------
! Procedures inside a Fortran MODULE have linker symbols of the form
!   __qlabs_MOD_qlabs_init     (GFortran convention)
! f2py expects the plain convention:
!   qlabs_init_
! This file provides plain standalone subroutines (not inside any module) that
! call through to the module versions — the same pattern used by warmdamp.f90
! (standalone) delegating to module ecdisp.
!
! The USE rename-only clause avoids a naming conflict at the Fortran source
! level:
!   use qlabs, only: qlabs_init_mod => qlabs_init
! renames the MODULE procedure to 'qlabs_init_mod' in this scope, while the
! subroutine being defined here is also called 'qlabs_init'.  At the linker
! level there is no conflict because the module procedure has symbol
! __qlabs_MOD_qlabs_init and this standalone subroutine has symbol qlabs_init_.
! ===========================================================================

subroutine qlabs_init(p_grid, xi_grid, psi_grid, edf_in, np, nxi, npsi)
  ! Load the LUKE EDF grid and initialise GL quadrature nodes/weights.
  ! Call once per WanKenoBeam run before any qlabs_absorption call.
  !
  ! Arguments:
  !   p_grid(np)               momentum grid, p/(m_e v_th)  [0, ∞)
  !   xi_grid(nxi)             pitch-angle cosine at Bmin   [-1, 1]
  !   psi_grid(npsi)           normalised poloidal flux     [0, 1]
  !   edf_in(np, nxi, npsi)    EDF f(p, xi, psi) from LUKE
  !   np, nxi, npsi            grid sizes

  use qlabs, only: qlabs_init_mod => qlabs_init
  implicit none

  integer,  intent(in) :: np, nxi, npsi
  real(8),  intent(in) :: p_grid(np)
  real(8),  intent(in) :: xi_grid(nxi)
  real(8),  intent(in) :: psi_grid(npsi)
  real(8),  intent(in) :: edf_in(np, nxi, npsi)

  call qlabs_init_mod(p_grid, xi_grid, psi_grid, edf_in, np, nxi, npsi)

end subroutine qlabs_init


subroutine qlabs_absorption(op, oc, Nr, theta, te, imod, psi_local, b_over_bmin, imNprw)
  ! Compute Im(N_perp) for the EDF loaded by qlabs_init.
  ! Interface is identical to warmdamp so that trace_one_ray.py needs only a
  ! one-line branch addition.
  !
  ! Inputs:
  !   op          = X  = omega_pe^2 / omega^2
  !   oc          = Y^2 = omega_ce^2 / omega^2
  !   Nr          = |N|  (total refractive index)
  !   theta       = angle between N and B  [rad]
  !   te          = electron temperature   [keV]  (bulk Maxwellian for warmdisp)
  !   imod        = +1 O-mode, -1 X-mode
  !   psi_local   = normalised poloidal flux at this ray point
  !   b_over_bmin = B_local / B_min  (magnetic trapping correction)
  ! Output:
  !   imNprw      = Im(N_perp)  [same sign convention as warmdamp]

  use qlabs, only: qlabs_absorption_mod => qlabs_absorption
  implicit none

  real(8),  intent(in)  :: op
  real(8),  intent(in)  :: oc
  real(8),  intent(in)  :: Nr
  real(8),  intent(in)  :: theta
  real(8),  intent(in)  :: te
  integer,  intent(in)  :: imod
  real(8),  intent(in)  :: psi_local
  real(8),  intent(in)  :: b_over_bmin
  real(8),  intent(out) :: imNprw

  call qlabs_absorption_mod(op, oc, Nr, theta, te, imod, psi_local, b_over_bmin, imNprw)

end subroutine qlabs_absorption
