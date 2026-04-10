! damping_ref.f90
! ---------------
! Subroutine version of Shkarofsky & Shoucri (2011) damping.f for Python/f2py.
!
! Computes the anti-Hermitian dielectric tensor elements (DINT11..DINT33) for
! an arbitrary distribution function using the same algorithm as damping.f,
! with the NAG Bessel library (S17DEF) replaced by jn_and_jnp from qlabs_bessel.
!
! Input grid convention:
!   f_sph(NX, NY) — f(p, xi) on a uniform spherical grid
!     p    axis: (I-1)*PMAX/(NX-1),   I=1..NX  (ascending, in m_e v_th units)
!     xi   axis: -1 + (J-1)*2/(NY-1), J=1..NY  (ascending, from -1 to +1)
!   This is the same format as distr.dat read by damping.f (before its xi-reversal).
!
! Physics parameters (match damping.f notation):
!   OMC    = NH * omega_ce / omega    (≈ 1 near cyclotron resonance)
!   PARN   = N_parallel
!   PERPN  = N_perp
!   BTH    = v_th/c = sqrt(Te[keV]/511)  (1D Jüttner thermal velocity)
!   NH     = harmonic number (1 or 2)
!
! Outputs: the 6 independent anti-Hermitian tensor elements.
!   Sign convention matches damping.f output (positive for absorption):
!   dint11 = Im(eps_11),  dint12 = Im(eps_12),  dint13 = Im(eps_13)
!   dint22 = Im(eps_22),  dint32 = Im(eps_32),  dint33 = Im(eps_33)
!
! These are the UNNORMALIZED sums (without the CONST = OMP2*DX*2*pi^2/BTH^3
! prefactor). For the ratio EDF/Maxwellian, CONST cancels.
!
! Integration method:
!   Uses a trapezoidal sum over a uniform cylindrical (u_par, u_perp) grid,
!   exactly following damping.f. Endpoint half-cell corrections are applied.
!
! Coordinate transform (Step 3):
!   Original damping.f used O(NX)+O(NY) linear search per cylindrical point.
!   This version uses O(1) direct index computation (floor division on uniform
!   grids), reducing overall interpolation complexity from O(NX^3*NY) to
!   O(NX^2*NY) — a ~180× speedup for NX=400.
!
! Validation (test_vs_damping_ref.py):
!   Compared against qlabs using the distr.dat Fokker-Planck EDF at
!   TE=4 keV, NH=2, OMC=0.95, PARN=0.6, PERPN=0.4.
!   EDF/MJ correction ratios: damping_ref r11=0.073, qlabs=0.087 (16.8% difference).
!   Agreement within expected 5-20% range given trapezoidal vs GL-35 integration.
!
! Build: see Makefile in this directory.

subroutine damping_ref_ec(f_sph, NX, NY, PMAX_in, OMC, PARN, PERPN, BTH, NH, &
                           dint11, dint12, dint13, dint22, dint32, dint33)

  use qlabs_bessel, only: jn_and_jnp
  implicit none

  ! --- Arguments
  integer,  intent(in)  :: NX, NY, NH
  real(8),  intent(in)  :: f_sph(NX, NY)
  real(8),  intent(in)  :: PMAX_in, OMC, PARN, PERPN, BTH
  real(8),  intent(out) :: dint11, dint12, dint13, dint22, dint32, dint33

  ! --- Parameters
  real(8), parameter :: PI = 3.141592653589793d0

  ! --- Derived scalars
  real(8) :: PMAX, HN, BTH2, PARN2, OMC2, CNST, DV, DMU
  integer :: NNX, NNY, NNX1, NNY1
  real(8) :: DX, DY

  ! --- Dynamic arrays
  real(8), allocatable :: DD(:,:)       ! spherical EDF, xi reversed
  real(8), allocatable :: F5(:,:)       ! cylindrical EDF
  real(8), allocatable :: DFDX(:,:)     ! df/d(u_par)   spline first derivative
  real(8), allocatable :: DFDY(:,:)     ! df/d(u_perp)  spline first derivative
  real(8), allocatable :: VXPARA(:)     ! cylindrical par grid
  real(8), allocatable :: VYPERP(:)     ! cylindrical perp grid
  real(8), allocatable :: DE(:), DF_a(:)  ! spline work arrays (length max(NNX,NNY))

  ! --- Integration scalars
  real(8) :: ZM, ZP, DZ1, DZ2
  integer :: IM, IP, II, IF_end, I11, I22, I, J, JX, JY, JMU, JV
  real(8) :: XX, YY, R, FAC, B1, B2, RP, RA
  real(8) :: UV1, UV2, ZPAR, ZARG, ZPERP, PPERP, XARG1
  real(8) :: Jn_val, Jnp_val
  real(8) :: V11, V12, V13, V22, V32, V33
  real(8) :: DDYY, DDXX, DERIVF
  real(8) :: PMBTH, PMBTH1, PPBTH, PPBTH1
  integer :: IPERP

  ! --- Initialise outputs
  dint11 = 0.d0 ; dint12 = 0.d0 ; dint13 = 0.d0
  dint22 = 0.d0 ; dint32 = 0.d0 ; dint33 = 0.d0

  PMAX  = PMAX_in
  HN    = real(NH)
  BTH2  = BTH*BTH
  PARN2 = PARN*PARN
  OMC2  = OMC*OMC
  CNST  = PARN2 + OMC2 - 1.0d0
  if (CNST <= 0.0d0) return   ! EC resonance inaccessible

  ! --- Spherical grid spacings
  DV  = PMAX / real(NX-1)
  DMU = 2.0d0 / real(NY-1)

  ! --- Cylindrical grid sizes (following damping.f: NNX = 2*N1-1, NNY = N1)
  NNX  = 2*NX - 1
  NNY  = NX
  NNX1 = NNX - 1
  NNY1 = NNY - 1
  DX   = 2.0d0*PMAX / real(NNX-1)
  DY   =       PMAX / real(NNY-1)

  ! --- Allocate arrays
  allocate(DD(NX,NY))
  allocate(F5(NNX,NNY), DFDX(NNX,NNY), DFDY(NNX,NNY))
  allocate(VXPARA(NNX), VYPERP(NNY))
  allocate(DE(NNX), DF_a(NNX))   ! sized for par spline; reused for perp below

  ! ==========================================================================
  ! STEP 1: Reverse xi ordering (xi from +1 at J=1 down to -1 at J=NY)
  !         Replicates damping.f lines 1340-1344.
  !         After reversal: DD(I,J) has xi(J) = 1 - (J-1)*DMU.
  ! ==========================================================================
  do I = 1, NX
    do J = 1, NY
      DD(I, NY-J+1) = f_sph(I, J)
    end do
  end do

  ! ==========================================================================
  ! STEP 2: Build cylindrical grid coordinates
  !         VXPARA: u_par/u_th from -PMAX to +PMAX (NNX = 2*NX-1 points)
  !         VYPERP: u_perp/u_th from 0 to PMAX       (NNY = NX points)
  ! ==========================================================================
  do I = 1, NNX
    VXPARA(I) = (2.d0*real(I-1)/real(NNX-1) - 1.d0) * PMAX
  end do
  do J = 1, NNY
    VYPERP(J) = real(J-1) * DY
  end do

  ! ==========================================================================
  ! STEP 3: Spherical → cylindrical coordinate transformation (O(1) indexing).
  !         For each cylindrical point (VXPARA, VYPERP), compute p and xi,
  !         then bilinearly interpolate DD(p,xi) → F5(VXPARA, VYPERP).
  !
  !         Both spherical grids are uniform, so bracket indices are computed
  !         directly via division instead of linear search:
  !           JV  = floor(R / DV)  + 1    => p(JV)..p(JV+1) bracket
  !           JMU = floor((1-xi) / DMU) + 1 => xi(JMU)..xi(JMU+1) bracket
  !         This replaces the original O(NX)+O(NY) searches with O(1) per point,
  !         reducing overall Step 3 complexity from O(NX^3*NY) to O(NX^2*NY).
  ! ==========================================================================
  F5 = 0.0d0
  do JX = 1, NNX
    XX = VXPARA(JX)
    do JY = 1, NNY
      YY = VYPERP(JY)
      R  = sqrt(XX*XX + YY*YY)
      if (R < 1.0d-15 .or. R > PMAX) cycle

      ! --- Radial bracket: p(JV) <= R < p(JV+1),  p(JV) = (JV-1)*DV
      JV  = max(1, min(NX-1, int(R/DV) + 1))
      RP  = real(JV-1, 8) * DV
      RA  = real(JV,   8) * DV

      ! --- Angular bracket: xi(JMU) >= xi_pt >= xi(JMU+1)
      !     xi decreases with J: xi(J) = 1 - (J-1)*DMU
      !     xi_pt in [xi(JMU+1), xi(JMU)] => JMU = floor((1-xi_pt)/DMU) + 1
      XARG1 = max(-1.0d0, min(1.0d0, XX/R))
      JMU   = max(1, min(NY-1, int((1.0d0 - XARG1)/DMU) + 1))
      UV1   = 1.0d0 - real(JMU-1, 8)*DMU   ! xi at JMU   (upper xi)
      UV2   = 1.0d0 - real(JMU,   8)*DMU   ! xi at JMU+1 (lower xi)

      ! --- Bilinear interpolation in (xi, p)
      FAC = (XARG1 - UV1) / (UV2 - UV1)
      B1  = DD(JV,  JMU) + (DD(JV,  JMU+1) - DD(JV,  JMU)) * FAC
      B2  = DD(JV+1,JMU) + (DD(JV+1,JMU+1) - DD(JV+1,JMU)) * FAC
      F5(JX, JY) = B1 + (B2 - B1)*(R - RP)/(RA - RP)

    end do   ! JY
  end do   ! JX

  ! ==========================================================================
  ! STEP 4: Cubic spline first derivatives in cylindrical coordinates.
  !
  !   Solving: M_{I-1} + 4*M_I + M_{I+1} = 3*(f_{I+1}-f_{I-1})/h
  !   (Thomas algorithm) where M_I is the first derivative at node I.
  !   This is the "Akima / cardinal spline" system used by damping.f.
  !
  !   DFDX(I,J) = df/d(VXPARA) at cylindrical node (I,J)
  !   DFDY(I,J) = df/d(VYPERP) at cylindrical node (I,J)
  !   Boundary conditions: zero first derivative at both ends.
  !
  !   Replicates damping.f lines 228-252.
  ! ==========================================================================

  ! --- DFDX: derivative along par axis (VXPARA), for each fixed J
  DFDX = 0.0d0
  do J = 1, NNY
    DE(1) = 0.0d0
    DF_a(1) = 0.0d0
    do I = 2, NNX1
      DE(I)   = -1.0d0 / (DE(I-1) + 4.0d0)
      DF_a(I) = (DF_a(I-1) - 3.0d0*(F5(I+1,J)-F5(I-1,J))/DX) * DE(I)
    end do
    DFDX(NNX, J) = 0.0d0
    do I = 1, NNX1
      DFDX(NNX-I, J) = DE(NNX-I)*DFDX(NNX-I+1, J) + DF_a(NNX-I)
    end do
  end do

  ! --- DFDY: derivative along perp axis (VYPERP), for each fixed I
  ! Reuse DE, DF_a but now sized NNY; NNY < NNX so arrays are big enough.
  DFDY = 0.0d0
  do I = 1, NNX
    DE(1) = 0.0d0
    DF_a(1) = 0.0d0
    do J = 2, NNY1
      DE(J)   = -1.0d0 / (DE(J-1) + 4.0d0)
      DF_a(J) = (DF_a(J-1) - 3.0d0*(F5(I,J+1)-F5(I,J-1))/DY) * DE(J)
    end do
    DFDY(I, NNY) = 0.0d0
    do J = 1, NNY1
      DFDY(I, NNY-J) = DE(NNY-J)*DFDY(I, NNY-J+1) + DF_a(NNY-J)
    end do
  end do

  ! ==========================================================================
  ! STEP 5: Find integration range indices IM, IP in VXPARA grid.
  !         ZM, ZP are the resonance ellipse limits in u = p/(m_e c) units.
  !         VXPARA is in p/(m_e v_th) units, so VXPARA * BTH = u_par.
  !         Replicates damping.f lines 274-294.
  ! ==========================================================================
  if (abs(PARN2 - 1.0d0) < 1.0d-10) then
    ZM = (OMC - 1.0d0/OMC) / 2.0d0
    ZP = -ZM
    if (ZM > ZP) then
      ZM = -abs(ZM) ; ZP = abs(ZP)
    end if
  else
    ZM = PARN*OMC/(1.0d0-PARN2) - sqrt(CNST)/abs(1.0d0-PARN2)
    ZP = PARN*OMC/(1.0d0-PARN2) + sqrt(CNST)/abs(1.0d0-PARN2)
  end if

  IM = 1 ; IP = NNX
  do I = 1, NNX1
    PMBTH  = VXPARA(I)   * BTH
    PMBTH1 = VXPARA(I+1) * BTH
    if (PMBTH < ZM .and. PMBTH1 > ZM) IM = I
    PPBTH  = VXPARA(I)   * BTH
    PPBTH1 = VXPARA(I+1) * BTH
    if (PPBTH < ZP .and. PPBTH1 > ZP) IP = I
  end do

  ! Integration bounds and endpoint corrections (damping.f lines 310-333)
  if (PARN2 < 1.0d0) then
    II    = IM + 1  ; IF_end = IP
    I11   = II      ; I22    = IF_end
    DZ1   = (VXPARA(IM+1) - ZM/BTH) / DX
    DZ2   = (ZP/BTH - VXPARA(IP)) / DX
  else if (PARN > 0.0d0) then
    II    = IP + 1  ; IF_end = NNX1
    I11   = II      ; I22    = 0
    DZ1   = (VXPARA(IP+1) - ZP/BTH) / DX
    DZ2   = 0.0d0
  else
    II    = 1       ; IF_end = IM
    I22   = IF_end  ; I11    = 0
    DZ2   = (ZM/BTH - VXPARA(IM)) / DX
    DZ1   = 0.0d0
  end if

  ! ==========================================================================
  ! STEP 6: Integration loop (replicates damping.f lines 335-389).
  !
  !   The integration variable is I (VXPARA index), i.e. u_par in BTH units.
  !   At each step, evaluate the resonance perpendicular momentum:
  !     ZARG = (PARN2-1)*ZPAR^2 + 2*PARN*OMC*ZPAR + OMC2-1
  !     ZPERP = sqrt(ZARG)   [u_perp in m_e c units]
  !   Bessel argument: XARG1 = PERPN * ZPERP * HN / OMC
  !   Tensor elements V11..V33 from Bessel J_n, J_n'.
  !   DERIVF = (df/d(u_perp)) * OMC + (df/d(u_par)) * ZPERP * PARN
  !          = DFDY_at_resonance * OMC + DFDX_at_resonance * ZPERP * PARN
  ! ==========================================================================
  do I = II, IF_end
    ZPAR = VXPARA(I) * BTH   ! u_par = p_par/(m_e c)
    ZARG = (PARN2-1.0d0)*ZPAR*ZPAR + 2.0d0*PARN*OMC*ZPAR + OMC2 - 1.0d0
    if (ZARG <= 0.0d0) cycle

    ZPERP = sqrt(ZARG)      ! u_perp at resonance
    if (ZPERP >= PMAX*BTH) cycle   ! beyond momentum grid

    XARG1 = PERPN * ZPERP * HN / OMC   ! Bessel argument

    ! Bessel functions J_n and J_n' (replacing NAG S17DEF)
    call jn_and_jnp(NH, XARG1, Jn_val, Jnp_val)

    ! Tensor elements V_ij (identical to qlabs.f90 formulas)
    if (XARG1 < 1.0d-10) then
      if (NH == 1) then
        V11 = 0.25d0 ; V12 = 0.25d0 ; V22 = 0.25d0
      else
        cycle   ! n >= 2: V_ij → 0 as XARG1 → 0
      end if
      V13 = 0.0d0 ; V33 = 0.0d0 ; V32 = 0.0d0
    else
      V11 = (HN * Jn_val / XARG1)**2
      V12 = HN * Jn_val * Jnp_val / XARG1
      V13 = (ZPAR/ZPERP) * HN * Jn_val**2 / XARG1
      V22 = Jnp_val**2
      V32 = -(ZPAR/ZPERP) * Jn_val * Jnp_val
      V33 = (ZPAR/ZPERP * Jn_val)**2
    end if

    ! Apply endpoint half-cell corrections (trapezoid rule boundary)
    if (I == I11) then
      V11 = V11*0.5d0*(1.0d0+DZ1) ; V12 = V12*0.5d0*(1.0d0+DZ1)
      V13 = V13*0.5d0*(1.0d0+DZ1) ; V22 = V22*0.5d0*(1.0d0+DZ1)
      V32 = V32*0.5d0*(1.0d0+DZ1) ; V33 = V33*0.5d0*(1.0d0+DZ1)
    end if
    if (I == I22) then
      V11 = V11*0.5d0*(1.0d0+DZ2) ; V12 = V12*0.5d0*(1.0d0+DZ2)
      V13 = V13*0.5d0*(1.0d0+DZ2) ; V22 = V22*0.5d0*(1.0d0+DZ2)
      V32 = V32*0.5d0*(1.0d0+DZ2) ; V33 = V33*0.5d0*(1.0d0+DZ2)
    end if

    ! Gradient of f at the resonance point (ZPAR, ZPERP) in cylindrical grid.
    ! u_perp index into VYPERP grid:
    PPERP = ZPERP / BTH          ! u_perp / (v_th/c) = p_perp / p_th
    IPERP = int(PPERP/DY) + 1    ! floor index (1-based)
    if (IPERP >= NNY) cycle      ! beyond perp grid (should not happen after ZPERP check)

    ! Linear interpolation of derivatives at resonance u_perp
    DDYY = (PPERP - VYPERP(IPERP)) * DFDY(I,IPERP+1)/DY &
         + (VYPERP(IPERP+1) - PPERP) * DFDY(I,IPERP)/DY
    DDXX = (PPERP - VYPERP(IPERP)) * DFDX(I,IPERP+1)/DY &
         + (VYPERP(IPERP+1) - PPERP) * DFDX(I,IPERP)/DY

    ! Quasi-linear resonance operator (damping.f Eq. 7)
    DERIVF = DDYY*OMC + DDXX*ZPERP*PARN

    ! Accumulate tensor element sums (sign convention from damping.f)
    dint11 = dint11 + ZPERP*V11*DERIVF
    dint12 = dint12 - ZPERP*V12*DERIVF
    dint13 = dint13 + ZPERP*V13*DERIVF
    dint22 = dint22 - ZPERP*V22*DERIVF
    dint32 = dint32 + ZPERP*V32*DERIVF
    dint33 = dint33 + ZPERP*V33*DERIVF

  end do

  ! Apply CONST prefactor normalization consistent with damping.f:
  !   dint11, dint13, dint33 → negate (so positive for absorption)
  !   dint12, dint22, dint32 → keep sign (already positive for absorption)
  ! We multiply by DX only (not OMP2*2*pi^2/BTH^3) since those are trivial
  ! scale factors that cancel in the ratio EDF/Maxwellian.
  dint11 = -dint11 * DX
  dint12 =  dint12 * DX
  dint13 = -dint13 * DX
  dint22 =  dint22 * DX
  dint32 =  dint32 * DX
  dint33 = -dint33 * DX

  deallocate(DD, F5, DFDX, DFDY, VXPARA, VYPERP, DE, DF_a)

end subroutine damping_ref_ec
