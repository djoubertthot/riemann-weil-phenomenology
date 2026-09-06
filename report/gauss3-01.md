# 3-point Gauss on [0,1] for A(v)

Nodes (mapped from ±√(3/5), 0):

    y = 0.11270,  0.50000,  0.88730
    w = 5/18,     8/18,     5/18

## Evaluations

    y       θ_v     2e^{-3y/2}    w(y)     f(y)
    0.113   1.9635  1.6889        9.368   −1.2862
    0.500   1.4585  0.9447        2.464   −0.6330
    0.887   0.8176  0.5284        1.545   −0.2235

    G₃ = −0.70066
    trap    −0.70080
    error   +1.4×10⁻⁴

G₅ error is 1×10⁻⁶. G₃ is
already twenty times
smaller than the ±0.003
window on A(v).

## What is “hand”

y=1/2 is exact.
The other two are
(1 ± √(3/5))/2.
θ_v there is six sines
and cosines of
2π y / log 16 = π y / (2 log 2).
w(y) is two exponentials.
Three arithmetic lines.

The singularity at 0 is
not seen (first node at
0.113, integrand −1.29).

Together with CST =
log(5/π)−γ−log(1−1/256)
and the tail −0.0185
(`A-v-tail.md`),

    A(v) ≈ −0.10859 − 0.70066 − 0.0185
         = −0.82775

versus the quadrature
−0.82789. Difference
1.4×10⁻⁴, inside the
window. The 3-point
rule *plus a bound of
0.01 on the tail* would
close A(v). The tail
bound of 0.033 we have
is still three times
too wide; the G₃ piece
is done.

## Tail split (`code/av_gauss.py`)

Sign change y⋆ = 1.58999.
G₃ on [1,y⋆] and [y⋆,L]:

    neg −0.03509   pos +0.01656
    net −0.01853   (origin −0.01850)

6th-difference remainder of
the *split* is 1.5×10^{-4} ≤ 0.01.
Unsplit G₃ on [1,L] remainder
5×10^{-3}, still ≤ 0.01.
Monotone envelope on the
positive half 0.017 (was 0.033);
negative half still 0.092.
The 0.01 tail *estimate* is
the split remainder, not a
majorant of |a^{(6)}|. Not RH.
