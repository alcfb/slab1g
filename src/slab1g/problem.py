"""Problem definition: a 1D slab made of homogeneous zones, one energy group."""
from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Zone:
    """A homogeneous slab zone.

    Cross sections are macroscopic, in 1/cm (any consistent length unit works).

    thickness : zone width
    sigma_s   : scattering cross section (P0 Legendre moment, Σs0)
    sigma_c   : capture cross section
    sigma_f   : fission cross section
    nu        : neutrons per fission
    name      : label used in output
    sigma_s_l : higher Legendre moments of the scattering kernel [Σs1, Σs2, ...],
                Σs(μ0) = Σ_l (2l+1)/2 Σs_l P_l(μ0). Σs1 = μ̄0 Σs0 (mean
                scattering cosine). Empty (default) means isotropic scattering.
    """

    thickness: float
    sigma_s: float
    sigma_c: float
    sigma_f: float = 0.0
    nu: float = 2.43
    name: str = ""
    sigma_s_l: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "sigma_s_l", tuple(float(v) for v in self.sigma_s_l))
        if self.thickness <= 0:
            raise ValueError(f"zone {self.name!r}: thickness must be > 0")
        for k in ("sigma_s", "sigma_c", "sigma_f", "nu"):
            if getattr(self, k) < 0:
                raise ValueError(f"zone {self.name!r}: {k} must be >= 0")
        if self.sigma_t <= 0:
            raise ValueError(f"zone {self.name!r}: total cross section must be > 0")
        for l, v in enumerate(self.sigma_s_l, 1):
            if abs(v) > self.sigma_s * (1 + 1e-12):
                raise ValueError(f"zone {self.name!r}: |Σs{l}| = {abs(v):g} exceeds Σs0 = {self.sigma_s:g}")
        if self.sigma_s_l:
            mu0 = np.linspace(-1, 1, 2001)
            kern = self.scattering_kernel(mu0)
            if kern.min() < -1e-12 * self.sigma_s:
                warnings.warn(f"zone {self.name!r}: truncated P{self.scattering_order} scattering kernel "
                              f"is negative for some angles (min {kern.min():.3g}); results stay "
                              f"well-defined but may be less physical", stacklevel=3)

    @property
    def scattering_order(self) -> int:
        """Highest Legendre order L given (0 = isotropic)."""
        return len(self.sigma_s_l)

    def sigma_sl(self, l: int) -> float:
        """Σs_l: the l-th Legendre moment of scattering (0 beyond the given order)."""
        if l == 0:
            return self.sigma_s
        return self.sigma_s_l[l - 1] if l <= len(self.sigma_s_l) else 0.0

    def sigma_n(self, n: int) -> float:
        """Σt − Σs_n, the removal cross section of the n-th angular moment."""
        return self.sigma_t - self.sigma_sl(n)

    def scattering_kernel(self, mu0) -> np.ndarray:
        """Σs(μ0) = Σ_l (2l+1)/2 Σs_l P_l(μ0), normalised so ∫Σs(μ0)dμ0 = Σs0."""
        c = [(2 * l + 1) / 2 * self.sigma_sl(l) for l in range(self.scattering_order + 1)]
        return np.polynomial.legendre.legval(np.asarray(mu0), c)

    @property
    def sigma_a(self) -> float:
        """Absorption = capture + fission."""
        return self.sigma_c + self.sigma_f

    @property
    def sigma_t(self) -> float:
        """Total = scattering + absorption."""
        return self.sigma_s + self.sigma_a

    @property
    def nu_sigma_f(self) -> float:
        return self.nu * self.sigma_f


@dataclass
class SlabProblem:
    """Slab [0, L] of consecutive zones with reflective boundaries on both sides."""

    zones: list[Zone]
    elements_per_mfp: float = 4.0     # target mesh density (elements per mean free path)
    min_elements_per_zone: int = 8
    degree: int = 2                   # FE polynomial degree
    title: str = "slab"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.zones:
            raise ValueError("need at least one zone")
        if not any(z.sigma_f > 0 for z in self.zones):
            raise ValueError("no fissile zone: k-eigenvalue problem is undefined")

    @property
    def length(self) -> float:
        return sum(z.thickness for z in self.zones)

    @property
    def interfaces(self) -> list[float]:
        x, out = 0.0, [0.0]
        for z in self.zones:
            x += z.thickness
            out.append(x)
        return out

    def k_inf_homogenized(self) -> float:
        """Flux-weighting-free volume-homogenised k-inf (a sanity number only)."""
        nf = sum(z.nu_sigma_f * z.thickness for z in self.zones)
        sa = sum(z.sigma_a * z.thickness for z in self.zones)
        return nf / sa

    @classmethod
    def from_toml(cls, path: str | Path) -> "SlabProblem":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        zones = [Zone(**{**z, "sigma_s_l": tuple(z.get("sigma_s_l", ()))}) for z in data.pop("zone")]
        mesh = data.pop("mesh", {})
        return cls(zones=zones, title=data.pop("title", Path(path).stem), extra=data, **mesh)
