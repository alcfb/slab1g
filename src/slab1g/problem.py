"""Problem definition: a 1D slab made of homogeneous zones, one energy group."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Zone:
    """A homogeneous slab zone.

    Cross sections are macroscopic, in 1/cm (any consistent length unit works).

    thickness : zone width
    sigma_s   : isotropic scattering cross section
    sigma_c   : capture cross section
    sigma_f   : fission cross section
    nu        : neutrons per fission
    name      : label used in output
    """

    thickness: float
    sigma_s: float
    sigma_c: float
    sigma_f: float = 0.0
    nu: float = 2.43
    name: str = ""

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(f"zone {self.name!r}: thickness must be > 0")
        for k in ("sigma_s", "sigma_c", "sigma_f", "nu"):
            if getattr(self, k) < 0:
                raise ValueError(f"zone {self.name!r}: {k} must be >= 0")
        if self.sigma_t <= 0:
            raise ValueError(f"zone {self.name!r}: total cross section must be > 0")

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
        zones = [Zone(**z) for z in data.pop("zone")]
        mesh = data.pop("mesh", {})
        return cls(zones=zones, title=data.pop("title", Path(path).stem), extra=data, **mesh)
