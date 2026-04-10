# Implementation Plan: Fully-Relativistic Arbitrary-EDF Absorption Module (`qlabs`)

*Written: 2026-04-08. To be continued after review.*

---

## 1. Project Context

### 1.1 What WanKenoBeam does

WanKenoBeam is a Monte-Carlo ray-tracing code for electron-cyclotron (EC) beams in tokamak
plasmas. It traces many individual rays and bins their contributions onto spatial grids. Each
ray carries a wave-action (power-like scalar `Wfct`) that decays along the ray according to:

```
dWfct/dt = -2 γ Wfct
```

where `γ` is the local absorption rate in s⁻¹. The code also has a quasi-linear diffusion
module (`QL_diffusion/`) that post-processes the ray trajectories to compute the QL
wave-plasma interaction tensor, which is fed to a Fokker-Planck (FP) solver (LUKE) to
compute the electron distribution function (EDF).

### 1.2 The iterative coupling we want

Currently, both the ray propagation and the absorption use a weakly-relativistic Maxwellian
assumption. The goal of this branch (`Development/QLcoupling`) is an iterative loop:

```
WanKenoBeam (propagation + absorption)
        ↓   wave-power deposition on the Maxwellian → QL tensor
LUKE Fokker-Planck
        ↓   non-Maxwellian EDF f(p, ξ, ψ)
WanKenoBeam (propagation unchanged; absorption recalculated with actual f)
        ↓   ...
repeat until convergence
```

Propagation (the real part of the dispersion relation, i.e. ray trajectories) is
**unchanged** — it depends on the bulk plasma which remains well-approximated by a
Maxwellian. Only the **absorption** (the anti-Hermitian part of the dielectric tensor) must
be recalculated with the actual EDF.

---

## 2. Current Absorption Implementation

Full details are in [absorption_modules.md](absorption_modules.md). Summary:

### 2.1 Westerhof — `DAMPBQ` (`lib/westerino/westerino.f90`)

- Weakly relativistic (WR) approximation, valid for T ≪ 511 keV.
- Maxwellian distribution function assumed.
- Uses the plasma dispersion function Z(ζ) (Krivenski-Orefice 1983 WR tensor).
- Includes fundamental and 2nd harmonic.
- Self-consistent iteration for N⊥ (up to 10 iterations, tolerance 0.01).
- Returns `PNI = Im(N⊥)·sin(θ)`.
- Computes polarization vectors internally but does NOT return them.

### 2.2 Farina — `warmdisp` (`lib/ecdisp/ecdisp.f90`, wrapper `warmdamp.f90`)

- More accurate: uses the fully relativistic Maxwell-Jüttner distribution.
- Explicit Hermitian (dispersive) + anti-Hermitian (absorptive) tensor split.
- Hermitian part: 501-point numerical integration over relativistic momentum using `expei`.
- Anti-Hermitian part: evaluates Maxwell-Jüttner at the exact relativistic resonant momenta
  u∥_± = [N∥·nωce/ω ± √((nωce/ω)² − (1−N∥²))] / (1−N∥²).
- Iterative N⊥ solve (up to 100 iterations, tolerance 1e-4).
- Returns complex `anpr`; polarization vectors `ex, ey, ez` are also computed but thrown away
  by the `warmdamp` wrapper.
- The `warmdamp` wrapper hard-codes `iwarm=3` (full numerical path) and `lrm=3`.

### 2.3 How absorption enters `trace_one_ray.py`

In `__absorption_coefficient__` (line 158):
1. One of the two Fortran routines is called → `absImN = Im(N⊥)`.
2. The TORBEAM-style conversion:
   ```
   gamma = k0 * absImN * f * Vnorm * sinVb
   ```
   where `f = |4·Tr(M)/S·H|` is a scale factor from the dispersion matrix and
   `Vnorm·sinVb` is the group velocity projected perpendicular to B.
3. `gamma` is returned and used as `dWfct/dt = −2 γ Wfct`.

A new `absorptionModule == 2` option will slot into this same structure.

---

## 3. The Shkarofsky & Shoucri (2011) Paper

**Reference:** I.P. Shkarofsky and M. Shoucri,  
*"A numerical code for the calculation of relativistic electron cyclotron damping with an
arbitrary distribution function"*,  
Computer Physics Communications 182 (2011) 1507–1517.

### 3.1 Core physics

The anti-Hermitian (absorptive) elements of the fully-relativistic dielectric tensor can be
written as a **single 1D integral along the resonance curve**. For |N∥| < 1:

**Eq. (1) of the paper:**
```
a_ij = −i · (2π²ωpe²/ω²) · ∫_{z∥⁻}^{z∥⁺} dz∥  V_ij · U_res · f  |_{z⊥=z_res(z∥)}
```

The key variables:
- z∥ = p∥/(m_e v_th), z⊥ = p⊥/(m_e v_th)  — momenta normalized to thermal momentum
- BTH = v_th/c = 0.04424·√(Te[keV])        — relativistic thermal parameter
- γ = √(1 + z∥²·BTH² + z⊥²·BTH²)          — Lorentz factor
- nΩc/ω = n·ωce/ω                           — normalized cyclotron frequency

The resonance condition (relativistic):
```
γ − N∥·z∥ − n·Ωc/ω = 0
```
traces an **ellipse** in (z∥, z⊥) space. Along this ellipse:

**Eq. (2): resonance curve**
```
z_res(z∥) = [−(N∥²−1)z∥² + 2N∥z∥(nΩc/ω) + (nΩc/ω)² − 1]^(1/2) / (1−N∥²)
```
Only the portion where z_res(z∥) > 0 is physical. The integration limits are:

**Eq. (3): limits of integration**
```
z∥_± = [N∥·nΩc/ω ± √((nΩc/ω)² − (1−N∥²))] / (1−N∥²)
```
These limits exist only when `(nΩc/ω)² > (1−N∥²)`, i.e., the resonance is accessible.

### 3.2 The integrand

**Eq. (8):** The basis polarization vectors:
```
V_1 = n·J_n(x)/x
V_2 = i·dJ_n/dx
V_3 = (z∥/z_res)·J_n(x)
where x = N⊥·z_res·n / (nΩc/ω)   (Bessel argument)
```

The 6 independent tensor components come from V_i·V_j*:
```
V_11 = V_1², V_12 = V_1·V_2, V_13 = V_1·V_3,
V_22 = V_2², V_32 = V_3·V_2, V_33 = V_3²
```

The U_res factor (the quasi-linear operator applied to f):
```
U_res = nΩc/ω · ∂f/∂z⊥ + N∥ · ∂f/∂z∥   |_{z∥, z_res(z∥)}
```
This is the same QL gradient operator as in the YodaU code (see Section 5).

### 3.3 Cases for |N∥| ≥ 1

For |N∥| > 1 (e.g. electron Bernstein waves), Eq. (4) replaces Eq. (1) with slightly
different integration limits and z_res formula (Eq. 5). For N∥ = 1 exactly, the limit
Eqs. (32)-(33) apply. All cases are handled in `damping.f`.

Landau damping (n=0) is a degenerate case with only ε_22, ε_23/ε_32, ε_33 surviving,
computed via a simplified version of the same loop.

### 3.4 Analytic Maxwellian validation

For the Maxwell-Jüttner distribution, the integral can be evaluated analytically (Eqs. 11-38
of the paper) in terms of modified Bessel functions I_k and K_k. The code `damping.f` 
implements both the numerical integral (for arbitrary EDF) and the analytic Maxwellian result
(for validation), showing excellent agreement.

---

## 4. The Accompanying Code: `damping.f`

Located at `/home/devlamin/For_QL_coupling/damping.f` (FORTRAN 77, 553 lines).

### 4.1 Control parameters

```fortran
PARAMETER(ITEST=0, NH=2, NB=4)
TE    = 4.0 keV        ! background temperature
OMP2  = 1.0            ! ωpe²/ω² (everything normalized to this)
OMC   = 0.95           ! n·ωce/ω
PARN  = 0.6            ! N∥
PERPN = 0.4            ! N⊥
PMAX  = 30.0           ! max momentum (units of p_th), PMAX=15 for Maxwellian test
N1    = 400            ! number of z∥ grid points
N2    = 198            ! number of cos(θ) grid points for the input EDF
```

### 4.2 EDF input and coordinate transform (lines 76-225)

The EDF from a FP code arrives on a **spherical grid** (p, cosθ) of size N1×N2:
- `DD(I,J)` — EDF on (p, ξ) grid, where p = I·DV·BTH (not quite — actually normalized)
- `XCC(I) = (I-1)*DV` — momentum p (units of p_th)
- `YCC(J) = -1 + (J-1)*DMU` — pitch-angle cosine ξ

**Critical step (DO 5, lines 166-202)**: transform to **cylindrical coordinates** (p∥, p⊥):
```
p∥ = VXPARA(I) in [−PMAX·BTH, +PMAX·BTH]   (BTH factor gives physical units)
p⊥ = VYPERP(J) in [0, PMAX·BTH]
```
For each cylindrical grid point (p∥, p⊥):
- Compute r = √(p∥² + p⊥²), θ = arccos(p∥/r)
- Find the bounding (JMU, JV) on the spherical grid
- Bilinear interpolation → F5(I,J)

This transform is crucial: the Shkarofsky integral is in cylindrical coordinates, but FP
codes (like LUKE) output on spherical grids.

### 4.3 Derivative calculation (lines 227-253)

Cubic spline derivatives (Thomas algorithm) computed separately for each row/column:
```
DFDX(I,J) = ∂F5/∂p∥ at (VXPARA(I), VYPERP(J))
DFDY(I,J) = ∂F5/∂p⊥ at (VXPARA(I), VYPERP(J))
```

These are computed in physical cylindrical coordinates (p∥ in units of p_th·BTH, 
p⊥ in units of p_th·BTH).

### 4.4 Finding resonance limits (lines 274-334)

Computes z∥_± (called ZM, ZP in the code):
```fortran
ZM = PARN*OMC/(1−PARN²) − sqrt(CNST)/|1−PARN²|
ZP = PARN*OMC/(1−PARN²) + sqrt(CNST)/|1−PARN²|
where CNST = N∥² + (nΩc/ω)² − 1
```
Then finds grid indices IM, IP bracketing ZM/BTH and ZP/BTH. Endpoint corrections DZ1, DZ2
are half-integer offsets to account for the fact that ZM and ZP don't fall exactly on the
grid.

### 4.5 Main integration loop: DO 600 (lines 336-390)

For each z∥ = VXPARA(I)·BTH (in the range [ZM, ZP]):
1. Compute `ZARG = (N∥²-1)·z∥² + 2N∥z∥·OMC + OMC² − 1`; skip if ≤ 0.
2. `ZPERP = sqrt(ZARG)` — resonant perpendicular momentum (times BTH).
3. Bessel argument: `XARG1 = N⊥·ZPERP·n/OMC`.
4. Call NAG `S17DEF(FNU, Z, NB)` for J_n, J_{n+1}, J_{n+2}, J_{n+3} via complex argument.
5. Compute V11, V12, V13, V22, V32, V33 (the tensor basis terms).
6. Bilinear interpolation: look up `DFDX`, `DFDY` at (p∥ = VXPARA(I), p⊥ = ZPERP/BTH).
7. `DERIVF = DDYY·OMC + DDXX·ZPERP·PARN`  — the quasi-linear gradient:
   ```
   DERIVF = ∂f/∂p⊥ · nΩc/ω + ∂f/∂p∥ · p⊥_res · N∥
   ```
   This is exactly `U_res` from the paper (Eq. 8), expressed in Shkarofsky's units.
8. Accumulate: `DINT11 += ZPERP · V11 · DERIVF` (and similarly for other components).
9. Half-weight correction at endpoint grid cells (DZ1, DZ2 factors).

**Note on integration scheme**: The code uses a simple summation (effectively trapezoidal
rule) over a **uniform** p∥ grid with step DX = PMAX·BTH·2/(N1−1). The endpoint correction
(lines 366-381) applies a `0.5*(1+DZ)` weight at the first/last cell to account for the
non-exact landing of the integration limit on the grid. This is a first-order scheme.

### 4.6 Normalization and output (lines 391-411)

```fortran
CONST = OMP2 * DX * 2π² / BTH³
DINT11 = −DINT11 * CONST
```

The `1/BTH³` factor comes from converting from (z∥, z⊥) units (normalized to p_th) back to
physical units. The result is Im(ε_11) normalized to ωpe²/ω².

Output: 6 independent components Im(ε_11), Im(ε_12), Im(ε_13), Im(ε_22), Im(ε_32), Im(ε_33).

---

## 5. Connection to YodaU (`radiative_transfer.py`)

The YodaU code (`YodaU/Modules/radiative_transfer.py`) implements the same physics for the
absorption coefficient α_ω in the radiative transfer equation. The connection is:

| Shkarofsky | YodaU |
|------------|-------|
| Integration variable z∥ (uniform grid) | Integration variable u_⊥ (Gauss-Legendre) |
| Resonant z_res(z∥) computed from ellipse | Resonant u∥_res(u_⊥) from resonance condition |
| V_ij = V_i·V_j (tensor components) | A_n = \|e·V\|² (scalar for specific polarization) |
| U_res = QL gradient of f | C[f] = (n/ω̄/u_⊥)·∂f/∂u_⊥ + N∥·∂f/∂u∥ |
| Returns Im(ε_ij) (full tensor) | Returns α_ω (absorption coeff for given wave mode) |
| EDF: cylindrical grid f(p∥, p⊥) | EDF: (p, ξ, ψ) spherical grid via `EDF_Interpolation` |
| NAG Bessel functions | `scipy.special.jv` / `jvp` |
| Simple trapezoidal sum on uniform grid | GL-48 on each sub-interval, split at singularity |

**The two are equivalent**: both compute the same physical quantity (the resonance curve
integral) just with different choices of:
- Integration variable (z∥ vs u_⊥)
- Quadrature scheme (trapezoidal on uniform grid vs GL on optimal nodes)
- Output format (full tensor vs scalar for specific wave mode)

**Key YodaU formula** (Absorption.abs_coef):
```
α = −2π²(ωpe²/cω) · ∫ A_n(u_⊥) · C[f](u_⊥, u∥_res) · u_⊥/γ  du_⊥
```
where A_n = |(ex + ω̄·N⊥/n · u∥·ez)·J_n(b) − (i·b/n)·J_n'(b)·ey|²  (polarization kernel).

This α_ω is the power absorption per unit path length (cm⁻¹). YodaU then uses it directly
in the radiative transfer equation dI/ds = −α·I. The singularity (where the resonance
denominator |u∥/γ − N∥| → 0) is handled by splitting the interval at the singular u_⊥ and
applying GL separately on each half.

**Key difference from what WanKenoBEAM needs**: WanKenoBEAM uses `gamma` (s⁻¹), not α_ω
(cm⁻¹). The conversion is `alpha = 2·gamma / |V_group|`, or equivalently
`gamma = alpha · |V_group| / 2`.

---

## 6. LUKE EDF Format

The FP code LUKE outputs the EDF on a **spherical momentum grid**:
```
f(p, ξ, ψ)    where:
  p   = |p| / (m_e v_th)    — total normalized momentum (≥ 0)
  ξ   = p∥ / |p|            — pitch-angle cosine at Bmin (∈ [−1, 1])
  ψ   = normalized poloidal flux (flux surface label)
```

- The ξ grid is defined at the **minimum B** point of the flux surface (bounce-averaged).
- The p and ξ grids can be non-uniform in the version used by WanKenoBEAM.
- Grid dimensions are typically O(400) in p × O(200) in ξ × O(50) in ψ.

**Coordinate transform to cylindrical** (needed for Shkarofsky integral):
```
p∥  = p · ξ
p⊥  = p · √(1 − ξ²)
```
Chain rule for derivatives:
```
∂f/∂p∥ = ∂f/∂p · ξ + ∂f/∂ξ · (1−ξ²)/p
∂f/∂p⊥ = √(1−ξ²) · [∂f/∂p − ξ/p · ∂f/∂ξ]
```

**Trapped-particle complication**: at a poloidal location where B/Bmin > 1, some electrons
are magnetically trapped and do not contribute to the local absorption. The accessible pitch
angles satisfy:
```
1 − (B/Bmin)·(1−ξ²) ≥ 0   →   |ξ_local| ≥ √(1 − Bmin/B)
```
This is handled in YodaU via `mhu_hxr` (the local pitch-angle grid). In WanKenoBEAM, the
ray already knows B at every point; the trapping boundary must be applied when evaluating
the EDF in the new module.

---

## 7. Implementation Plan

### 7.1 Module location and structure

```
WanKenoBEAM/RayTracing/lib/qlabs/
├── qlabs.f90          — main Fortran 90 module (public interface)
├── qlabs_edf.f90      — EDF storage, coordinate transform, interpolation + derivatives
├── qlabs_bessel.f90   — Bessel J_n evaluation (replace NAG dependency)
├── Makefile           — builds libqlabs.a; follows ecdisp/Makefile pattern
└── qlabs.pyf          — f2py interface; follows warmdamp.pyf pattern
```

`qlabs` will be a peer of `ecdisp` and `westerino` in `RayTracing/lib/`.

### 7.2 Public subroutine interface

**Subroutine 1 — Initialization (called once per WanKenoBEAM run):**
```fortran
subroutine qlabs_init(p_grid, xi_grid, psi_grid, edf, &
                       np, nxi, npsi)
  ! Stores the EDF on its native (p, xi, psi) spherical grid
  ! Precomputes derivatives ∂f/∂p and ∂f/∂xi using cubic splines
  ! (same Thomas-algorithm splines as damping.f)
  ! Does NOT transform to cylindrical yet (too large; done per-call)
  integer, intent(in) :: np, nxi, npsi
  real(r8), intent(in) :: p_grid(np), xi_grid(nxi), psi_grid(npsi)
  real(r8), intent(in) :: edf(np, nxi, npsi)
end subroutine
```

**Subroutine 2 — Per-point absorption (called once per ray-point where absorption is active):**
```fortran
subroutine qlabs_absorption(xg, yg, anpl, anprc, sox, psi_local, &
                              B_over_Bmin, absImN)
  ! xg     = ωpe²/ω²
  ! yg     = ωce/ω
  ! anpl   = N∥
  ! anprc  = initial guess N⊥ (cold solution, for warmdisp)
  ! sox    = wave mode (-1=O, +1=X)
  ! psi_local = poloidal flux label at this ray point
  ! B_over_Bmin = B/Bmin at this poloidal position (for trapping)
  ! absImN = Im(N⊥)  [same output as warmdamp]
  real(r8), intent(in)  :: xg, yg, anpl, anprc, psi_local, B_over_Bmin
  integer,  intent(in)  :: sox
  real(r8), intent(out) :: absImN
end subroutine
```

### 7.3 Core algorithm inside `qlabs_absorption`

**Step A — Get polarization and real N⊥ from Farina (unchanged):**
```fortran
call warmdisp(xg, yg, anpl, amu_bulk, sox, iwarm=3, lrm=3, anprc, &
              anpr_real, ex, ey, ez, ierr)
! anpr_real = real part of N⊥  (discard imaginary part from Maxwellian)
! ex, ey, ez = complex polarization unit vector
N_perp_real = real(anpr_real)
```
The `amu_bulk` is computed from the **background temperature** T_e (used only for the
Hermitian part / real N⊥). This is justified: the bulk Maxwellian controls wave propagation;
the tail EDF modifies only the absorption.

**Step B — Compute resonance limits:**
```fortran
CNST = anpl**2 + yg**2 - 1.0_r8        ! = N∥² + (Ωc/ω)² − 1
if (CNST <= 0.0_r8) then
    absImN = 0.0_r8 ; return            ! resonance inaccessible
end if
ZM = anpl*yg/(1.0_r8 - anpl**2) - sqrt(CNST)/abs(1.0_r8 - anpl**2)
ZP = anpl*yg/(1.0_r8 - anpl**2) + sqrt(CNST)/abs(1.0_r8 - anpl**2)
```
For the N∥ ≥ 1 case, use the alternative Eqs. (5)-(6) from the paper.

**Step C — Determine trapping cutoff:**
The accessible pitch angle range at B/Bmin:
```fortran
xi_trap = sqrt(max(0.0_r8, 1.0_r8 - 1.0_r8/B_over_Bmin))
! Electrons with |xi| < xi_trap are trapped and don't contribute
```
Any quadrature point where the resonant pitch angle |ξ_res| < xi_trap is excluded.

**Step D — Gauss-Legendre quadrature over p∥ ∈ [ZM, ZP]:**

Using n_gl = 35–48 GL nodes (to be determined by convergence testing):
```fortran
call gauleg(ZM, ZP, z_gl_nodes, w_gl, n_gl)   ! GL nodes+weights on [ZM,ZP]
integral = 0.0_r8
do i = 1, n_gl
    zpar  = z_gl_nodes(i)                        ! z∥ node
    ZARG  = (anpl**2-1)*zpar**2 + 2*anpl*zpar*yg + yg**2 - 1
    if (ZARG <= 0.0_r8) cycle                    ! skip if off resonance ellipse
    zperp = sqrt(ZARG)                           ! z⊥ on resonance curve
    
    ! --- Evaluate EDF and derivatives at this resonance point ---
    ppar_phys  = zpar  * BTH                     ! p∥ in p_th units
    pperp_phys = zperp * BTH                     ! p⊥ in p_th units
    p_sph      = sqrt(ppar_phys**2 + pperp_phys**2)
    xi_sph     = ppar_phys / p_sph
    
    ! Check trapping
    if (abs(xi_sph) < xi_trap) cycle
    
    ! Interpolate f and its derivatives from (p,xi,psi) grid
    call qlabs_edf_eval(p_sph, xi_sph, psi_local, f_val, dfdp, dfdxi)
    
    ! Convert to cylindrical derivatives via chain rule
    dfdpar  = dfdp * xi_sph + dfdxi * (1 - xi_sph**2) / p_sph
    dfdperp = sqrt(1-xi_sph**2) * (dfdp - xi_sph/p_sph * dfdxi)
    
    ! Quasi-linear operator = U_res (Shkarofsky Eq. 8)
    DERIVF = dfdperp * yg + dfdpar * zperp * anpl
    
    ! --- Compute polarization kernel ---
    ! Bessel argument
    xarg = N_perp_real * zperp / yg        ! = N⊥ · z⊥/Ωc
    call qlabs_bessel_jn(n_harm, xarg, Jn, Jnp)   ! J_n and J_n'
    
    ! Projection A_n = |e · V|² (YodaU-style scalar for this wave mode)
    V1 = real(n_harm) * Jn / xarg
    V2 = Jnp
    V3 = (zpar/zperp) * Jn
    e_dot_V = ex*V1 + ey*V2 + ez*V3       ! complex scalar
    A_n = abs(e_dot_V)**2
    
    ! Accumulate integrand (Jacobian zperp from Shkarofsky Eq. 1)
    gamma_res = sqrt(1.0_r8 + (zpar**2 + zperp**2)*BTH**2)
    integral = integral + w_gl(i) * A_n * DERIVF * zperp / gamma_res
end do
```

**Alternatively (and possibly cleaner)**: compute all 6 tensor components (as in `damping.f`)
and then contract with the polarization vector to get Im(ε_eff). This approach requires
fewer decisions about the Bessel argument normalization and is easier to validate against
the Shkarofsky analytical Maxwellian results.

**Step E — Convert to Im(N⊥):**

Two equivalent approaches:

*Approach E1 — Direct α_ω and back-conversion:*
```
α_ω = −OMP2 * CONST * integral   (with CONST = 2π²/BTH³)
gamma = α_ω * |V_group| / 2
absImN = gamma / (k0 * f_scale * Vnorm * sinVb)
```
This requires knowing `f_scale`, `Vnorm`, `sinVb` inside the Fortran module (or passing
them in), which pollutes the interface with TORBEAM-specific quantities.

*Approach E2 — Weak damping formula (recommended):*
From the polarization eigenvector and the cold dispersion relation:
```
Im(ε_eff) = e*ᵢ · Im(εᵢⱼ) · eⱼ     (computed from tensor components above)
Im(N⊥) = −Im(ε_eff) / (∂D_cold/∂N⊥²) / (2·N⊥_real)
```
where `∂D_cold/∂N⊥²` can be computed analytically from the cold Appleton-Hartree
dispersion relation at the operating point (xg, yg, anpl, N⊥_real). This is purely
algebraic and requires no additional Fortran calls.

*Approach E3 — Pass back Im(ε_eff) and let Python do the conversion:*
The Python layer in `trace_one_ray.py` already computes `f`, `Vnorm`, `sinVb`. The
module could return Im(ε_eff) directly, and Python converts to absImN via the same
formulas. This is the least intrusive change to the Fortran module but requires modifying
`trace_one_ray.py` more substantially.

**Recommendation: E2** — cleanest, self-contained, returns the same `absImN` float as
the existing `warmdamp`, requires no changes to the Python calling code beyond adding an
`absorptionModule == 2` branch.

### 7.4 EDF interpolation (`qlabs_edf.f90`)

The EDF module stores:
- `p_grid(np)`, `xi_grid(nxi)`, `psi_grid(npsi)` — 1D grids
- `edf(np, nxi, npsi)` — the EDF values
- `dedf_dp(np, nxi, npsi)`, `dedf_dxi(np, nxi, npsi)` — pre-computed spline derivatives

Per-point evaluation (called from Step D above):
1. Binary search in psi_grid, p_grid, xi_grid for the cell containing (p, ξ, ψ).
2. Trilinear interpolation for f_val.
3. Derivatives: use pre-computed spline coefficients (Thomas algorithm, same as damping.f).

**Note on log-EDF**: For numerical stability, store and interpolate log(f) (as YodaU does
in `EDF_Interpolation`). The derivatives of f = exp(log f) are then:
```fortran
dfdp  = f_val * d_logf_dp
dfdxi = f_val * d_logf_dxi
```
This prevents underflow for the low-density EDF tail and is consistent with the YodaU
approach.

### 7.5 Bessel functions (`qlabs_bessel.f90`)

`damping.f` uses NAG routines (S17DEF, S18DCF, S18DEF). Replace with:
- Option A: Implement J_n and J_n' from the `ecdisp` module — it already calls `jv` via
  the Faddeeva/ssbi infrastructure. But those are for integer orders.
- Option B: Use the Miller downward recurrence for J_n with real arguments:
  - Start from high order, recurse downward, normalize via ∑J_{2k} = 1.
  - Accurate for real arguments and n ≤ 5 (the typical EC harmonic range).
- **Recommended: Option B** — simple, self-contained, well-established.

For the Bessel argument `x = N⊥·z⊥·n/(nΩc/ω)`: when x is small (near perpendicular
propagation), use the small-argument expansion `J_n(x) ≈ (x/2)ⁿ/n!` (as `damping.f` does
with `CY22`).

### 7.6 Integration quadrature choice

The original `damping.f` uses a uniform p∥ grid with N1=400 points (effectively
trapezoidal rule + endpoint correction). This is accurate but potentially slow.

For the new module, Gauss-Legendre quadrature is preferred because:
1. The integrand (EDF × Bessel) is smooth within the resonance interval.
2. GL-35 gives the same accuracy as trapezoid-400 for smooth integrands.
3. Reduces the number of expensive EDF interpolations from 400 to 35.

**Singularity handling**: The integrand DERIVF = ∂f/∂p⊥·Ωc + ∂f/∂p∥·p⊥·N∥ can have a
formal singularity where z_res → 0 (for very small z∥ near the integration limit). This
is handled by the integration limits themselves (the resonance ellipse ends where z_res = 0,
which is z∥ = z∥_±). The GL nodes near the endpoints automatically have small weight, so
no explicit singularity treatment is needed unless z_res → 0 inside the interval (which
can happen for N∥ close to ±1). For |N∥| < 0.99, this is not an issue.

An alternative (matching YodaU more closely): integrate over z⊥ (the perpendicular
coordinate) with z∥_res(z⊥) determined by the resonance condition. This flips the role of
the integration variable but gives the same result and handles the near-perpendicular case
more naturally. Decision to be made during implementation.

---

## 8. Integration into WanKenoBeam

### 8.1 Python-level changes (`trace_one_ray.py`)

In `__init__`:
```python
if self.absorptionModule == 2:
    from RayTracing.lib.qlabs.qlabsECabsorption import qlabs_init, qlabs_absorption
    qlabs_init(p_grid, xi_grid, psi_grid, edf, ...)   # load LUKE EDF once
```

In `__absorption_coefficient__`:
```python
if self.absorptionModule == 2:
    absImN = qlabs_absorption(parAlpha, sqrt(parBeta), Nparallel, Nnorm,
                               sigma, psi_local, B_over_Bmin)
```
The EDF is loaded once at startup and stored in module-level Fortran arrays (using Fortran
`save` arrays or module variables), so subsequent calls are cheap.

### 8.2 Input file changes

New input parameter: `absorptionModule = 2` (existing 0/1 unchanged).
Also required: path to LUKE output file containing EDF, p-grid, xi-grid, psi-grid.

### 8.3 The outer iterative loop

The outer loop (WanKenoBeam ↔ LUKE) will be orchestrated at the Python/notebook level:
1. Run WanKenoBeam with `absorptionModule = 1` (Farina Maxwellian) → get QL tensor.
2. Run LUKE → get non-Maxwellian EDF.
3. Re-run WanKenoBeam with `absorptionModule = 2` + new EDF → updated QL tensor.
4. Repeat until converged.

This outer loop logic is out of scope for this branch but should be designed with in mind.

---

## 9. Open Technical Questions

**Q1. Coordinate frame for the EDF at local B/Bmin.**
The LUKE EDF is bounce-averaged (ξ at Bmin). For the absorption at a specific poloidal
position where B ≠ Bmin, the effective local pitch angle ξ_local = sign(ξ)·√(1 − (B/Bmin)·(1−ξ²)).
The YodaU code handles this explicitly. Should the new module accept the full (p, ξ_Bmin) EDF
and do this transform internally, or expect a pre-transformed (p∥_local, p⊥_local) EDF?

**Recommendation**: Accept the native LUKE format (p, ξ_Bmin, ψ) and handle the trapping/
mirror mapping internally, passing B/Bmin as an argument. This is more general and consistent
with how LUKE outputs data.

**Q2. Which tensor components to compute.**
`damping.f` computes all 6 independent components. For WanKenoBEAM's absorption coefficient,
only the scalar projection e*·Im(ε)·e is needed. Computing the full tensor is more general
(useful for debugging and future applications such as EC emission in a future ECE
synthetic diagnostic mode) but more expensive (factor ~6).

**Recommendation**: Implement both: a full-tensor version (for validation against
Shkarofsky analytical results) and a projection-only version (for production use in ray
tracing).

**Q3. Harmonic summation.**
`damping.f` computes a single harmonic NH. WanKenoBEAM also operates at a single harmonic
(determined by the EC wave frequency and the local magnetic field). The `warmdamp` wrapper
uses `larmornumber` to find the dominant harmonic. This logic should be replicated in
`qlabs_absorption`.

For plasma parameters where two neighboring harmonics contribute (large N∥ or high
harmonic number), the code should sum over n and n+1. This is flagged in the paper but
not implemented in `damping.f`. For ECRH on TCV (and similar machines), a single harmonic
is sufficient.

**Q4. The Hermitian part of ε (propagation vs. absorption).**
The paper says: "We have only addressed the anti-Hermitian components of the dielectric
tensor (which determine damping or emission). The Hermitian parts (which determine
propagation) are usually determined by the bulk, which can generally be approximated by a
relativistic Maxwellian."

This is exactly the approach here: Farina's `warmdisp` provides Re(N⊥) and the polarization
from the bulk Maxwellian, and the new module provides Im(N⊥) from the non-Maxwellian EDF.
The background temperature T_e (for the Maxwellian Hermitian part) should be the bulk
temperature from the equilibrium, not inferred from the EDF tail.

**Q5. The conversion from Im(ε_eff) to Im(N⊥) — Approach E2 details.**
The cold dispersion relation for an O or X mode has the form:
```
D_cold(N⊥²) = A·N⊥⁴ + B·N⊥² + C = 0
```
with A, B, C functions of xg, yg, N∥. The derivative:
```
∂D_cold/∂N⊥² = 2A·N⊥²_real + B
```
This is purely algebraic, known, and already implicitly computed in `warmdisp`. In practice,
∂D/∂N⊥ can be read from the TORBEAM-style quantities already available in `trace_one_ray.py`.
The cleanest implementation may be to pass the relevant group-velocity quantities from Python
into `qlabs_absorption` to avoid re-deriving them.

---

## 10. Performance

Expected call frequency: ~100 ray-points × 10,000 rays × n_iterations.
Per-call cost breakdown:
- `warmdisp` (Farina, for polarization): ~100 μs (existing, unchanged)
- EDF interpolation (trilinear, 3D): ~5 μs per GL node × 35 nodes = ~175 μs
- Bessel function evaluation: ~2 μs per node × 35 = ~70 μs
- Total per call: ~350 μs vs ~100 μs for current warmdamp

Expected slowdown: ~3–4× per absorbed ray-step. Since rays are only slow in the absorption
region and the absorption calculation dominates that region already, the practical simulation
time increase is perhaps 2–3× overall — acceptable.

**Optimization levers if needed**:
1. Reduce GL points from 35 to 20 (test convergence first).
2. Pre-compute Bessel functions at GL nodes for the current N⊥ and cache across nearby ray steps.
3. Use coarser EDF grid near the boundaries of momentum space.
4. Parallelize: each ray is already an independent MPI/OpenMP task in WanKenoBEAM.

---

## 11. Build System

Following the `ecdisp` pattern:
```makefile
# lib/qlabs/Makefile
F90 = gfortran
FFLAGS = -O3 -fPIC -ffree-form

SRCS = qlabs_edf.f90 qlabs_bessel.f90 qlabs.f90
OBJS = $(SRCS:.f90=.o)

libqlabs.a: $(OBJS)
    ar rcs $@ $^

# Link against libecdisp.a for warmdisp
qlabs.o: qlabs.f90
    $(F90) $(FFLAGS) -I../ecdisp -c $< -o $@
```

f2py wrapper (similar to `warmdamp.pyf`):
```python
# In qlabs.pyf:
python module qlabsECabsorption
  interface
    subroutine qlabs_init(p_grid, xi_grid, psi_grid, edf, np, nxi, npsi)
      intent(in) :: p_grid, xi_grid, psi_grid, edf, np, nxi, npsi
    end subroutine
    subroutine qlabs_absorption(xg, yg, anpl, anprc, sox, psi_local, &
                                 B_over_Bmin, absImN)
      intent(in)  :: xg, yg, anpl, anprc, sox, psi_local, B_over_Bmin
      intent(out) :: absImN
    end subroutine
  end interface
end python module
```

---

## 12. Validation Plan

1. **Maxwellian recovery test** (`ITEST=1` equivalent):  
   Run `qlabs_absorption` with the Maxwell-Jüttner EDF (same one as Farina uses internally).
   Result should match `warmdamp` output to within a few percent.

2. **Shkarofsky analytical comparison**:  
   For a Maxwellian EDF, compare against the analytical Eqs. (14)–(19) of the paper.
   The existing `damping.f` already demonstrates this — replicate in the new code.

3. **YodaU comparison**:  
   For a non-Maxwellian EDF loaded from `distr.dat` (the example file in
   `/home/devlamin/For_QL_coupling/distr.dat`), compare `qlabs` output against a
   YodaU `abs_coef` call with the same EDF, N∥, N⊥, ω values.

4. **Convergence test**:  
   Vary the number of GL nodes from 10 to 100; confirm that 35 nodes gives < 1% error
   relative to 100 nodes for the test EDF.

5. **Self-consistency test**:  
   After the first WanKenoBEAM + LUKE iteration, verify that the QL tensor from
   `QL_diffusion` computed with `absorptionModule=2` is consistent with the LUKE EDF
   that was used as input.

---

## 13. Files Created in This Session

- [absorption_modules.md](absorption_modules.md) — detailed explanation of Westerhof and
  Farina absorption modules (the two currently implemented routines).
- [QLcoupling_plan.md](QLcoupling_plan.md) — this document.

---

## 14. References

1. Shkarofsky & Shoucri (2011), CPC 182, 1507–1517.
   `/home/devlamin/For_QL_coupling/` contains the paper PDF, `damping.f`, and `distr.dat`.

2. Farina, D. — `ecdisp` module in `RayTracing/lib/ecdisp/ecdisp.f90`.  
   Computes relativistic warm-plasma dielectric tensor; source of `warmdisp` subroutine.

3. Westerhof, E. — `westerino` module in `RayTracing/lib/westerino/westerino.f90`.  
   Weakly-relativistic WR absorption via Krivenski-Orefice plasma dispersion functions.

4. YodaU project at `/home/devlamin/YodaU/`.  
   Synthetic ECE diagnostic; absorption in `Modules/radiative_transfer.py` (same physics
   as target new module, implemented in Python with GL quadrature over u_⊥).

5. LUKE Fokker-Planck code — EDF output on (p, ξ, ψ) spherical grid.  
   Reference interface: `YodaU/Modules/edf_interpolation.py` (`EDF_Interpolation` class).

6. Krivenski & Orefice (1983), J. Plasma Phys. 30, 125.  
   Basis for the weakly-relativistic dielectric tensor in Westerhof's `EPSILON` subroutine.

7. Decker & Ram (2006), Phys. Plasmas 13, 112503.  
   EC and EBW damping in non-Maxwellian plasmas; validates the anti-Hermitian tensor approach.
