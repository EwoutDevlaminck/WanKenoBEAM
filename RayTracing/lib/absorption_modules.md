# EC Absorption Modules: Westerhof (`DAMPBQ`) and Farina (`warmdisp`)

This document describes the two absorption routines currently used in WanKenoBeam, explains
exactly what physics each one computes, and discusses what the new fully-general module needs
to replace.

---

## 1. Westerhof — `DAMPBQ` (`westerino/westerino.f90`)

### Interface

```fortran
subroutine DAMPBQ(PTHETA, PNR, PNI, PALFA, PBETA, PVTE, PMODE, ICALLED)
```

| Argument   | Direction | Meaning |
|------------|-----------|---------|
| `PTHETA`   | in        | Angle between wave vector N and B [rad] |
| `PNR`      | in        | Real part of refractive index N (cold N⊥ as initial guess) |
| `PNI`      | out       | Im(N⊥) · sin(θ), the absorptive imaginary part |
| `PALFA`    | in        | X = ωpe²/ω² (density parameter) |
| `PBETA`    | in        | Y² = ωce²/ω² (magnetic-field parameter) |
| `PVTE`     | in        | vte/c (thermal velocity normalised to c) |
| `PMODE`    | in        | +1 = O-mode, −1 = X-mode |
| `ICALLED`  | in        | 0 = initialise (first call on ray), 1 = continue |

### Physical model

Weakly relativistic (WR) approximation with a Maxwellian distribution function.
Includes first AND second cyclotron harmonics.

### Step-by-step algorithm

**Step 1 — Parameter setup.**
From `PVTE` and the input angles:
```
ZMU   = 2/VTE²      (≡ mec²/kBTe in WR limit, up to a factor 2)
N∥    = PNR·cos(θ)
N⊥    = PNR·sin(θ)
```
`ZMU` plays the role of the inverse temperature: larger = colder plasma.

**Step 2 — Warm dielectric tensor via `DIELTE → EPSILON`.**

`EPSILON` (Krivenski & Orefice 1983, J. Plasma Phys. 30, 125) computes the weakly-relativistic
dielectric tensor as a sum over Larmor harmonics n = −1, 0, +1, +2. For each harmonic it
evaluates the velocity-space integrals through the **plasma dispersion function**

```
Z(ζ) = i√π · w(ζ)     (w = Faddeeva function)
```

implemented in the local function `ZETA(Z)` using Watanabe's algorithm. The key argument is

```
ζ_± = ψ ± φ,    ψ = N∥·√(μ/2),    φ = √(μ·α),
α = N∥²/2 − 1 + n·ωce/ω
```

`α > 0` means the resonance is accessible; the imaginary part of `Z(ζ)` then produces a
non-zero contribution to the absorptive (anti-Hermitian) part of the tensor. This is where the
"weakly relativistic" approximation enters: the resonance condition is evaluated as
`γ ≈ 1 + u²/2` rather than exactly, which is only valid for `vte/c ≪ 1`.

The general-purpose (anti-)loss-cone distribution is also supported via the integer parameters
`JNPAR`, `JNPER` passed to `EPSILON`; in the standard WanKenoBeam call these are both 0
(pure Maxwellian).

**Step 3 — Biquadratic dispersion equation.**

With the tensor in hand, `DAMPBQ` forms the wave-equation determinant

```
D(N⊥²) = CA · N⊥⁴ + CB · N⊥² + CC = 0
```

where `CA`, `CB`, `CC` are built from the tensor elements after separating N∥ and N⊥:

```
CA  = ΛXZ² + ΛXX · ΛZZ
CB  = − ... (from tensor Λ = ε − N²1 + NN)
CC  = ε₃₃⁽⁰⁾ · (εXY² + ΛXX · ΛYY)
```

The quadratic is solved for N⊥² and the correct root is picked at the first call by choosing
whichever has its real part closest to the cold N⊥. A branch-cut tracker (`NSIGN`, `NQUAD`)
then maintains continuity along the ray for all subsequent calls (`ICALLED=1`).

**Step 4 — Self-consistent iteration.**

Because the tensor itself depends on the current N⊥ (through the `N⊥` entries in `EPSILON`),
the whole process repeats up to `MAXIT = 10` times until `|N⊥_new − N⊥_old| < 0.01`.

**Step 5 — Output.**

```
PNI = Im(N⊥) · sin(θ)
```

Note: the code also computes polarization vectors `(ex, ey, ez)` for both O and X modes
(the Bertelli 2010 addition near line 275), but **these are not returned through the
interface** — they were added for diagnostic purposes and sit unused in the wrapper.

### What this routine assumes

- The electron velocity distribution is an isotropic Maxwellian.
- Weak relativistic correction: the resonance denominator is expanded to second order in vte/c.
- The harmonic structure is correct through the 2nd harmonic.
- No sub-thermal features (loss cone, bump-on-tail) unless `JNPAR/JNPER` are set.

---

## 2. Farina — `warmdisp` / `warmdamp` (`ecdisp/ecdisp.f90` + `ecdisp/warmdamp.f90`)

### Interface (inner routine called by the wrapper)

```fortran
subroutine warmdisp(xg, yg, anpl, amu, sox, iwarm, lrm, anprc, anpr, ex, ey, ez, ierr)
```

| Argument | Direction | Meaning |
|----------|-----------|---------|
| `xg`     | in   | X = ωpe²/ω² |
| `yg`     | in   | Y = ωce/ω |
| `anpl`   | in   | N∥ |
| `amu`    | in   | μ = mec²/(kBTe) = 511/Te[keV] (relativistic temperature parameter) |
| `sox`    | in   | Wave mode: −1 = O, +1 = X (sign convention reversed vs. Westerhof) |
| `iwarm`  | in   | Approximation level: ≤2 → WR polynomial; >2 → fully relativistic (FR) |
| `lrm`    | in   | Maximum Larmor harmonic included |
| `anprc`  | in   | Initial guess for N⊥ (cold solution) |
| `anpr`   | out  | Complex N⊥; Im(N⊥) encodes absorption |
| `ex,ey,ez` | out | Complex polarization unit vector |
| `ierr`   | out  | Error flag (99 = cutoff, 100 = iteration not converged) |

The wrapper `warmdamp.f90` calls this with `iwarm = 3` (fully relativistic) and `lrm = 3`
(up to 3rd harmonic), then returns only `imNprw = aimag(anpr)`, discarding `ex,ey,ez`.

### Physical model

Fully relativistic Maxwell-Jüttner distribution function; arbitrary Larmor harmonic order.
The key improvement over Westerhof is the correct relativistic resonance condition

```
γ − N∥·u∥ − n·ωce/ω = 0
```

evaluated exactly without the WR expansion.

### Step-by-step algorithm

**Step 1 — Dielectric tensor via `dieltens_maxw_fr`.**

The tensor is split into Hermitian (H) and anti-Hermitian (AH) parts computed separately.

*Hermitian part* — `hermitian(yg, anpl, amu, rr, lrm, iwarm)`:

For `iwarm > 2` (fully relativistic):
- A 501-point uniform grid `ttv(i)` covering [−5, 5] is pre-computed in `set_extv` with
  weights `extdtv(i) = exp(−t²)·dt`.
- The integration variable maps to relativistic parallel momentum via:
  ```
  u∥(t) = bth·t·sqrt(1 + t²/(2μ)),    γ(t) = 1 + t²/μ
  ```
  (bth = √(2/μ) is the thermal momentum).
- For each grid point and each harmonic n, the integrand involves the exponential integral
  `expei(x) = exp(−x)·Ei(x)` where `x = −μ·(γ − n·ωce/ω − N∥·u∥)`.
  This evaluates the **Cauchy principal value** of the resonance denominator,
  i.e. the dispersive (non-absorptive) part.
- The result `rr(n,m,l)` arrays contain the moments needed to assemble all nine tensor elements
  at each Larmor order `l` and moment order `m`.

For `iwarm ≤ 2` (weakly relativistic):
- Overwrites the numerical result with explicit polynomial expansions in
  `bth² = 2/μ = 2kBTe/mec²`, valid to 8th order in the thermal parameter.
  These are the Krivenski-Orefice-style analytic WR integrals.

*Anti-Hermitian part* — `antihermitian(yg, anpl, amu, ri, lrm)`:

This computes the δ-function residues at the resonant momenta. The fully relativistic
resonance condition for harmonic n

```
γ − N∥·u∥ = n·ωce/ω
```

gives two resonant parallel momenta:

```
u∥_± = (N∥·n·Y ± √((n·Y)² − (1−N∥²))) / (1−N∥²)
```

where Y = ωce/ω. These only exist (resonance is accessible) when
`(nY)² > 1 − N∥²`.

At each resonant point, the Maxwell-Jüttner distribution is evaluated and the perpendicular
velocity moment integrals `ri(n,m,l)` = `∫ u⊥^(2m) f_MJ · δ(resonance) du⊥` are computed.
Two regimes:
- `|aa| = |μ·N∥·Δu∥| > 5`: asymptotic expansion via recursion on fi0, fi1, fi2.
- `|aa| ≤ 5`: uses modified spherical Bessel functions via `ssbi(aa, n, lrm)`.

**Step 2 — Assembly of tensor per Larmor order `l`.**

H and AH parts are combined as

```
ε(l)_jk = −xg·(H_jk + i·AH_jk)·fal
```

where `fal = (−1/4)^l · (2l)! / (l!)² / Y^(2l−2)` is the Larmor expansion coefficient.
The full tensor is then a power series in N⊥²:

```
ε_jk(N⊥²) = Σ_{l=1}^{lrm} ε(l)_jk · (N⊥²)^(l−1)
```

**Step 3 — Iterative dispersion relation in `warmdisp`.**

Same biquadratic structure as Westerhof but the coefficients are now evaluated self-consistently
with the FR dielectric tensor:

```
cc4·N⊥⁴ + cc2·N⊥² + cc0 = 0
cc4 = (ε11−N∥²)(1−a33) + (a13+N∥)(a31+N∥)
cc0 = ε330·((ε11−N∥²)(ε22−N∥²) + ε12²)
```

where `a13, a23, a33` are the N⊥-dependent off-diagonal elements. Iteration proceeds
up to 100 times with tolerance `|1 − |N⊥²_new/N⊥²_old|| < 1e-4`.

**Step 4 — Polarization extraction.**

From the converged N⊥ and the assembled tensor:
```
ey = −(ε12·(ε13+N⊥·N∥) + (ε11−N∥²)·ε23) / D
ez = ( ε12² + (ε22−N⊥²−N∥²)·(ε11−N∥²)) / D
D  = ε12·ε23 − (ε13+N⊥·N∥)·(ε22−N⊥²−N∥²)
ex = 1 / sqrt(1 + |ey|² + |ez|²),   then ey *= ex,  ez *= ex
```

These are the Jones-vector components of the electric field in the (x,y,z) frame aligned to B
and the propagation plane.

### What this routine assumes

- The distribution is a relativistic Maxwellian (Maxwell-Jüttner).
- For `iwarm > 2` (the mode used by `warmdamp`): no WR approximation in the anti-Hermitian
  part; the resonance is located exactly.
- The Hermitian part is still evaluated on a fixed 501-point grid — accurate but not exact.
- The distribution is isotropic (no pitch-angle structure, no suprathermal component).

---

## 3. What Both Routines Return to WanKenoBeam

Both are called inside `trace_one_ray.py:__absorption_coefficient__` and return a single scalar:

| Routine | Returns | Used as |
|---------|---------|---------|
| `DAMPBQ` | `PNI = Im(N⊥)·sin(θ)` | → `absImN = PNI/sin(θ)` |
| `warmdamp` | `imNprw = Im(N⊥)` | → `absImN = imNprw` (directly) |

Both are then fed into the TORBEAM-style formula:

```
γ = k₀ · absImN · f · |Vg| · sin(Vg∧B)
```

where `f` is the dispersion-matrix scale factor and `|Vg|·sin(Vg∧B)` is the group velocity
component perpendicular to B.

---

## 4. Language Recommendation for the New Module

The new module must evaluate the general absorption integral

```
α = −2π² · (ωpe²/(c·ω)) · Σ_n ∫₀^{p_max} A_n(u⊥,u∥_res) · C[f](u∥_res,u⊥) · u⊥/γ du⊥
```

where `A_n` contains the Bessel-function polarization kernel (from `ex,ey,ez` and N⊥) and
`C[f] = (n/ω̄/u⊥)·∂f/∂u⊥ + N∥·∂f/∂u∥` is the quasi-linear operator on the arbitrary EDF.

### Fortran — recommended

**Reasons to prefer Fortran:**

1. **Direct reuse of `warmdisp`.**  The polarization vectors `ex, ey, ez` are already computed
   inside `warmdisp` as an intermediate result. A thin Fortran wrapper can call `warmdisp`
   (unchanged) and receive `(anpr, ex, ey, ez)` immediately. C++ calling Fortran requires
   explicit `extern "C"` declarations, careful name mangling (`warmdisp_` vs `warmdisp__`
   depending on compiler), and manual type-matching — fragile and adds a maintenance burden.

2. **The integration is nested loops over u⊥ and harmonics** — exactly the kind of code
   Fortran compiles to optimal SIMD/vectorised instructions with no extra effort. A Gauss-Legendre
   quadrature over 48–64 points per harmonic in double-precision complex arithmetic is idiomatic
   Fortran.

3. **The EDF interpolation is trilinear on a 3D grid** (p, ξ, ψ). This is three nested binary
   searches and eight multiply-adds. Trivial in Fortran, and the grid arrays passed in from Python
   via f2py are already contiguous Fortran-order arrays.

4. **Build system already in place.** The `ecdisp` and `westerino` libraries are both built with
   `Makefile` + `f2py`. Adding a third library with the same pattern requires almost no tooling
   change. f2py interface files (`.pyf`) are straightforward to write and the compiler flags are
   already established.

5. **Code coherence.** The three absorption modules (`westerino`, `ecdisp`, new one) become
   peers in `RayTracing/lib/`. The Python layer in `trace_one_ray.py` imports all three
   identically via their f2py wrappers. A C++ extension compiled via pybind11 would introduce
   a heterogeneous build dependency and a different calling convention.

6. **Modern Fortran (2003+) is sufficient.** Array syntax, allocatables, and modules provide
   everything needed. There is no need for C++ templates or RAII for a single-purpose
   quadrature kernel.

**The only real argument for C++** would be if you later needed the module to be called
from a C or Julia environment, or if you wanted pybind11's more ergonomic Python interface.
Neither applies here — f2py is already the interface layer and works well.

### Summary

Write the new module as a Fortran 90 module in `RayTracing/lib/qlabs/qlabs.f90`, structured
symmetrically to `ecdisp/warmdamp.f90`:

- Call `warmdisp` (from `ecdisp`) internally to get `(anpr, ex, ey, ez)`.
- Accept the EDF as a flat array plus grid metadata (passed from Python each call, or better,
  stored in a module-level saved array after a one-time initialisation call).
- Return `absImN` in exactly the same form as `warmdamp`, so that the calling code in
  `trace_one_ray.py` requires only the addition of `absorptionModule == 2`.
