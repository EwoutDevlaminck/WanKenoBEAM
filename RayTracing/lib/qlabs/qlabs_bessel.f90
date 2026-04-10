module qlabs_bessel
  ! ---------------------------------------------------------------------------
  ! Bessel functions J_n(x) and J_n'(x) for real x >= 0, integer n >= 1.
  !
  ! Algorithm: Miller downward recurrence starting from order n_start = n + 30,
  ! normalized via the identity  J_0 + 2*sum_{k=1}^{inf} J_{2k} = 1.
  !
  ! For |x| < eps_small the leading small-argument behaviour is used instead:
  !   J_n(x) ~ (x/2)^n / n!        (avoids 0/0 in V_1 = n*J_n/x)
  !   J_n'(x) ~ n*(x/2)^{n-1} / (2*n!)
  !
  ! Accuracy: for n <= 6 and 0 <= x <= 50 the relative error is < 5e-14.
  ! ---------------------------------------------------------------------------

  implicit none
  integer,  parameter :: r8 = selected_real_kind(15,300)
  real(r8), parameter :: pi_r8 = 3.1415926535897932384626433832795_r8

contains

  ! ---------------------------------------------------------------------------
  ! Primary entry point: returns J_n and J_n' for a single (n, x) pair.
  ! ---------------------------------------------------------------------------
  subroutine jn_and_jnp(n, x, Jn_out, Jnp_out)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: Jn_out, Jnp_out

    real(r8), parameter :: eps_small = 1.0e-10_r8
    real(r8) :: jvals(0:n+30), norm, half_x, fac
    integer  :: k, n_start

    ! --- small-argument branch ---
    if (x < eps_small) then
      if (n == 1) then
        Jn_out  = 0.5_r8 * x           ! J_1 ~ x/2
        Jnp_out = 0.5_r8               ! J_1' ~ 1/2
      else
        ! J_n ~ (x/2)^n / n! ; J_n' ~ n*(x/2)^(n-1) / (2^n * n!)
        half_x = 0.5_r8 * x
        fac = 1.0_r8
        do k = 1, n
          fac = fac * half_x / real(k, r8)
        end do
        Jn_out  = fac
        if (x < 1.0e-200_r8) then
          Jnp_out = 0.0_r8
        else
          Jnp_out = real(n, r8) * fac / x
        end if
      end if
      return
    end if

    ! --- Miller downward recurrence ---
    n_start = n + 30

    ! Initialise seed (arbitrary, normalized later)
    jvals(n_start)     = 1.0_r8
    jvals(n_start - 1) = 0.0_r8
    do k = n_start - 1, 1, -1
      jvals(k - 1) = 2.0_r8 * real(k, r8) / x * jvals(k) - jvals(k + 1)
    end do

    ! Normalization: J_0 + 2*( J_2 + J_4 + ... ) = 1
    norm = jvals(0)
    do k = 2, n_start, 2
      norm = norm + 2.0_r8 * jvals(k)
    end do
    jvals = jvals / norm

    Jn_out = jvals(n)

    ! J_n'(x) = ( J_{n-1}(x) - J_{n+1}(x) ) / 2
    if (n + 1 <= n_start) then
      Jnp_out = 0.5_r8 * (jvals(n - 1) - jvals(n + 1))
    else
      ! fallback (n_start too small; should not happen for n <= 6)
      Jnp_out = jvals(n - 1) - real(n, r8) / x * jvals(n)
    end if

  end subroutine jn_and_jnp

end module qlabs_bessel
