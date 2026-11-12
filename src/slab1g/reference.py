"""Independent reference: diamond-difference discrete ordinates (S_N).

Classic finite-volume transport sweeps with power iteration, written without
nutils so it can cross-check the even-parity FE transport solver. Not meant to
be fast; use fine cells (DD is second-order in cell width).
"""
from __future__ import annotations

import numpy as np

from .problem import SlabProblem
from .solvers import gauss_legendre_half


def dd_sn(problem: SlabProblem, N: int = 16, cells_per_mfp: float = 20.0,
          tol: float = 1e-9, max_iter: int = 20000):
    """Return (k, cell centres, cell-average scalar flux, iterations).

    Anisotropic scattering is handled with the usual Legendre-moment source
    q(mu) = sum_l (2l+1)/2 Ss_l P_l(mu) phi_l, l <= min(L, N-1).
    """
    mu, w = gauss_legendre_half(N)
    L = min(max(z.scattering_order for z in problem.zones), N - 1)

    dx, st, ssl, nsf = [], [], [], []
    for z in problem.zones:
        nc = max(10, int(np.ceil(z.thickness * z.sigma_t * cells_per_mfp)))
        dx += [z.thickness / nc] * nc
        st += [z.sigma_t] * nc
        ssl += [[z.sigma_sl(l) for l in range(L + 1)]] * nc
        nsf += [z.nu_sigma_f] * nc
    dx, st, nsf = map(np.asarray, (dx, st, nsf))
    ssl = np.asarray(ssl).T                     # (L+1, nc)
    nc = len(dx)

    # P_l(mu_m) for the positive directions; P_l(-mu) = (-1)^l P_l(mu)
    P = np.array([np.polynomial.legendre.Legendre.basis(l)(mu) for l in range(L + 1)])  # (L+1, nmu)
    sgn = (-1.0) ** np.arange(L + 1)
    coef = (2 * np.arange(L + 1) + 1)[:, None] / 2 * ssl                                   # (L+1, nc)

    # DD coefficients per (direction, cell)
    tau = st[None, :] * dx[None, :] / mu[:, None]
    a = (1 - tau / 2) / (1 + tau / 2)
    b = (dx[None, :] / mu[:, None]) / (1 + tau / 2)

    phil = np.zeros((L + 1, nc))
    phil[0] = 1.0
    k = 1.0
    in_left = np.zeros(len(mu))    # psi(0, +mu): reflects psi(0, -mu)
    in_right = np.zeros(len(mu))   # psi(L, -mu): reflects psi(L, +mu)
    for it in range(max_iter):
        fis = nsf * phil[0] / (2 * k)
        qp = P.T @ (coef * phil) + fis[None, :]                    # (nmu, nc), directions +mu
        qm = (P * sgn[:, None]).T @ (coef * phil) + fis[None, :]   # directions -mu
        avg_p = np.empty((len(mu), nc))
        avg_m = np.empty((len(mu), nc))
        psi = in_left.copy()
        for i in range(nc):
            out = a[:, i] * psi + b[:, i] * qp[:, i]
            avg_p[:, i] = 0.5 * (psi + out)
            psi = out
        in_right = psi
        psi = in_right.copy()
        for i in range(nc - 1, -1, -1):
            out = a[:, i] * psi + b[:, i] * qm[:, i]
            avg_m[:, i] = 0.5 * (psi + out)
            psi = out
        in_left = psi
        new = (P * w[None, :]) @ avg_p + (P * (sgn[:, None] * w[None, :])) @ avg_m   # (L+1, nc)

        k_new = k * np.dot(nsf * dx, new[0]) / np.dot(nsf * dx, phil[0])
        err = max(abs(k_new - k), np.max(np.abs(new[0] / new[0].sum() - phil[0] / phil[0].sum())) * nc)
        phil, k = new, k_new
        if err < tol:
            break
    else:
        raise RuntimeError("DD S_N did not converge")

    x = np.concatenate([[0.0], np.cumsum(dx)])
    xc = 0.5 * (x[1:] + x[:-1])
    phi = phil[0] * problem.length / np.dot(phil[0], dx)
    return k, xc, phi, it + 1
