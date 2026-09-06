"""3-point Gauss of A(v) on [0, 1] for v = (4, −3, 1)/√26 at χ₅ μ=16.

θ_v is the elementary mix of the six lag kernels θ_nm, n,m ≤ 2
(H_2plane_independent.th). The A-integrand

    a(y) = (1/2) w(y) (2 e^{−3y/2} − θ_v(y)),
    w(y) = 2 e^{−y/2} / (1 − e^{−2y}),

is regular at 0: w ∼ 1/y is cancelled by 2 e^{−3y/2} − θ_v ∼ c y
with c = −3 − θ_v'(0). Origin named this Gauss check
(report/A-v-01.md). Not a covering lemma.

    python code/av_gauss.py
"""
from __future__ import annotations

import math

import numpy as np

# Rational witness, χ₅ μ=16.
MU = 16.0
L16 = math.log(MU)
S26 = math.sqrt(26.0)
V = (4.0 / S26, -3.0 / S26, 1.0 / S26)

# 3-point Gauss–Legendre on [0, 1] (origin: nodes ½ ± √(3/5)/2, ½).
_SQRT35 = math.sqrt(3.0 / 5.0)
GAUSS_NODES = (0.5 - 0.5 * _SQRT35, 0.5, 0.5 + 0.5 * _SQRT35)
GAUSS_WEIGHTS = (5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0)

# ∫_0^1 f = (1/2) ∫_{-1}^1 f((t+1)/2) dt. n=3 error constant times f^{(6)}.
# (1/2) * 2^{7} (3!)^4 / (7 (6!)^3) / 2^6 = (3!)^4 / (7 (6!)^3)
GAUSS3_REMAINDER_COEFF = (math.factorial(3) ** 4) / (
    7.0 * (math.factorial(6) ** 3)
)


def th(n, m, y, L):
    """Elementary lag table, same as H_2plane_independent.th (floats)."""
    om = lambda k: 2.0 * math.pi * k / L
    if n == 0 and m == 0:
        return 2.0 * (L - y) / L
    if n == 0 or m == 0:
        j = max(n, m)
        return -2.0 * math.sin(om(j) * y) / (math.sqrt(2.0) * math.pi * j)
    if n == m:
        return 2.0 * (
            (L - y) * math.cos(om(n) * y) / L
            - math.sin(om(n) * y) / (2.0 * math.pi * n)
        )
    return (
        2.0
        * (n * math.sin(om(n) * y) - m * math.sin(om(m) * y))
        / (math.pi * (m * m - n * n))
    )


def theta_v(y, L=L16, v=V):
    """θ_v(y) = ∑_{n,m=0}^{2} v_n v_m θ_nm(y). Six elementary functions."""
    y = float(y)
    if y <= 0.0:
        return 2.0 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if y >= L:
        return 0.0
    acc = 0.0
    for n in range(3):
        for m in range(3):
            acc += v[n] * v[m] * th(n, m, y, L)
    return acc


def theta_v_prime_0(L=L16, v=V):
    """θ_v'(0) from the differentiated lag table. c = −3 − this."""
    a, b, c = v
    invL = 1.0 / L
    s2 = math.sqrt(2.0)
    # θ00' = −2/L; θ0j'(0) = −2√2 / L (j=1,2);
    # θ11'(0)=θ22'(0)=−4/L; θ12'(0)=−4/L.
    return (
        a * a * (-2.0 * invL)
        + b * b * (-4.0 * invL)
        + c * c * (-4.0 * invL)
        + 2.0 * a * b * (-2.0 * s2 * invL)
        + 2.0 * a * c * (-2.0 * s2 * invL)
        + 2.0 * b * c * (-4.0 * invL)
    )


def kernel_limit_0(L=L16, v=V):
    """lim_{y→0} w(y)(2 e^{−3y/2} − θ_v(y)) = −3 − θ_v'(0)."""
    return -3.0 - theta_v_prime_0(L, v)


def w(y):
    if y <= 0.0:
        raise ValueError("w is 1/y at 0; use kernel_limit_0 / a(0)")
    return 2.0 * math.exp(-0.5 * y) / (1.0 - math.exp(-2.0 * y))


def kernel(y, L=L16, v=V):
    """w(y)(2 e^{−3y/2} − θ_v(y)), regularized at 0."""
    y = float(y)
    if y <= 1e-12:
        return kernel_limit_0(L, v)
    return w(y) * (2.0 * math.exp(-1.5 * y) - theta_v(y, L, v))


def a_integrand(y, L=L16, v=V):
    """Contribution to A(v): ½ kernel. ∫_0^1 a ≈ −0.70065 (origin)."""
    return 0.5 * kernel(y, L, v)


def gauss3_unit(f):
    """3-point Gauss–Legendre of f on [0, 1]."""
    return sum(w * f(x) for x, w in zip(GAUSS_NODES, GAUSS_WEIGHTS))


def max_abs_d6(f, a=0.0, b=1.0, n=240):
    """Max |Δ^6 f / h^6| of the shipped f on n+1 samples (6th difference)."""
    if n < 8:
        raise ValueError("need n >= 8")
    xs = np.linspace(a, b, n + 1)
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    h = (b - a) / n
    d6 = np.diff(ys, n=6) / (h ** 6)
    return float(np.max(np.abs(d6)))


def remainder_bound(f=None, a=0.0, b=1.0):
    """|E| ≤ (b-a)^7 * coeff × max|f^{(6)}| on [a, b]."""
    if f is None:
        f = a_integrand
    m = max_abs_d6(f, a, b)
    return (b - a) ** 7 * GAUSS3_REMAINDER_COEFF * m, m


def diff_ec_theta(y, L=L16, v=V):
    """2 e^{−3y/2} − θ_v(y). Zero near y=1.59 (origin)."""
    return 2.0 * math.exp(-1.5 * y) - theta_v(y, L, v)


def sign_change_y(L=L16, v=V, lo=1.0, n=60):
    """Unique root of 2 e^{−3y/2} − θ_v on (1, L), bisection."""
    a, b = lo, L
    fa, fb = diff_ec_theta(a, L, v), diff_ec_theta(b, L, v)
    if fa * fb > 0:
        raise ValueError("no sign change of 2e^{-3y/2}-θ_v on (1, L)")
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = diff_ec_theta(m, L, v)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def gauss3_interval(f, a, b):
    """3-point Gauss–Legendre of f on [a, b]."""
    scale = b - a
    return scale * sum(
        wt * f(a + scale * x) for x, wt in zip(GAUSS_NODES, GAUSS_WEIGHTS)
    )


def tail_comparison_bound(L=L16, v=V):
    """Monotone envelope: |diff| max at the left of each half, w decreasing.

    Origin's crude 0.033 was the positive half only. This returns
    (neg_bound, pos_bound, y_star) as absolute bounds on |∫ a|.
    """
    ys = sign_change_y(L, v)
    d1 = abs(diff_ec_theta(1.0, L, v))
    dL = abs(diff_ec_theta(L, L, v))
    w1 = w(1.0)
    wy = w(ys)
    neg = 0.5 * w1 * d1 * (ys - 1.0)
    pos = 0.5 * wy * dL * (L - ys)
    return neg, pos, ys


def tail_report(L=L16, v=V):
    ys = sign_change_y(L, v)
    g_neg = gauss3_interval(a_integrand, 1.0, ys)
    g_pos = gauss3_interval(a_integrand, ys, L)
    g_all = gauss3_interval(a_integrand, 1.0, L)
    r_neg, _ = remainder_bound(a_integrand, 1.0, ys)
    r_pos, _ = remainder_bound(a_integrand, ys, L)
    r_all, d6 = remainder_bound(a_integrand, 1.0, L)
    cneg, cpos, _ = tail_comparison_bound(L, v)
    return {
        "y_star": ys,
        "gauss_neg": g_neg,
        "gauss_pos": g_pos,
        "gauss_tail": g_all,
        "gauss_split": g_neg + g_pos,
        "rem_neg": r_neg,
        "rem_pos": r_pos,
        "rem_all": r_all,
        "d6_all": d6,
        "cmp_neg": cneg,
        "cmp_pos": cpos,
        "cmp_net": cpos - cneg,
    }


def report():
    lim = kernel_limit_0()
    g0 = a_integrand(0.0)
    gval = gauss3_unit(a_integrand)
    rem, d6 = remainder_bound()
    window = 0.003
    print(f"theta_v(0)={theta_v(0.0):.6f}  theta_v'(0)={theta_v_prime_0():.6f}")
    print(f"kernel_limit_0={lim:.6f}  (origin ≈ -2.962)")
    print(f"a(0)={g0:.6f}")
    print(f"gauss3[0,1] a = {gval:.6f}  (origin I_[0,1] ≈ -0.70065)")
    print(f"nodes={[round(x, 6) for x in GAUSS_NODES]}")
    print(f"|a^{(6)}|_max ≈ {d6:.4e}  remainder_bound={rem:.4e}")
    if rem > window:
        print(f"remainder {rem:.4e} exceeds ±{window} A-window; Gauss does not close Q(v)>0 by hand")
    else:
        print(f"remainder {rem:.4e} sits inside ±{window} A-window")
    t = tail_report()
    print(f"y_star={t['y_star']:.6f}")
    print(f"gauss3[1,L] a = {t['gauss_tail']:.6f}  split={t['gauss_split']:.6f}  (origin tail ≈ -0.01850)")
    print(f"gauss neg/pos = {t['gauss_neg']:.6f} / {t['gauss_pos']:.6f}")
    print(f"rem[1,L]={t['rem_all']:.4e}  rem split={t['rem_neg']+t['rem_pos']:.4e}")
    print(f"cmp |neg|/|pos| = {t['cmp_neg']:.4f} / {t['cmp_pos']:.4f}  (origin crude pos 0.033)")
    target = 0.01
    split_rem = t["rem_neg"] + t["rem_pos"]
    if split_rem <= target:
        print(f"split remainder {split_rem:.4e} ≤ {target} tail target")
    else:
        print(f"split remainder {split_rem:.4e} still above {target} tail target")
    return gval, lim, rem, t


if __name__ == "__main__":
    report()
