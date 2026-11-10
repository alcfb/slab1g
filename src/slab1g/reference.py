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
    mu, w = gauss_legendre_half(N)

    dx, st, ss, nsf = [], [], [], []
    for z in problem.zones:
        nc = max(10, int(np.ceil(z.thickness * z.sigma_t * cells_per_mfp)))
        dx += [z.thickness / nc] * nc
        st += [z.sigma_t] * nc
        ss += [z.sigma_s] * nc
        nsf += [z.nu_sigma_f] * nc
    dx, st, ss, nsf = map(np.asarray, (dx, st, ss, nsf))
    nc = len(dx)

    # DD coefficients per (direction, cell)
    tau = st[None, :] * dx[None, :] / mu[:, None]
    a = (1 - tau / 2) / (1 + tau / 2)
    b = (dx[None, :] / mu[:, None]) / (1 + tau / 2)

    phi = np.ones(nc)
    k = 1.0
    in_left = np.zeros(len(mu))    # psi(0, +mu): reflects psi(0, -mu)
    in_right = np.zeros(len(mu))   # psi(L, -mu): reflects psi(L, +mu)
    for it in range(max_iter):
        q = 0.5 * (ss * phi + nsf * phi / k)          # isotropic emission per unit mu
        phi_new = np.zeros(nc)
        # sweep +mu
        psi = in_left.copy()
        for i in range(nc):
            out = a[:, i] * psi + b[:, i] * q[i]
            phi_new[i] += np.dot(w, 0.5 * (psi + out))
            psi = out
        in_right = psi
        # sweep -mu
        psi = in_right.copy()
        for i in range(nc - 1, -1, -1):
            out = a[:, i] * psi + b[:, i] * q[i]
            phi_new[i] += np.dot(w, 0.5 * (psi + out))
            psi = out
        in_left = psi

        k_new = k * np.dot(nsf * dx, phi_new) / np.dot(nsf * dx, phi)
        err = max(abs(k_new - k), np.max(np.abs(phi_new / phi_new.sum() - phi / phi.sum())) * nc)
        phi, k = phi_new, k_new
        if err < tol:
            break
    else:
        raise RuntimeError("DD S_N did not converge")

    x = np.concatenate([[0.0], np.cumsum(dx)])
    xc = 0.5 * (x[1:] + x[:-1])
    phi = phi * problem.length / np.dot(phi, dx)
    return k, xc, phi, it + 1
