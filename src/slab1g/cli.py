"""Command line: solve a slab with diffusion, SP3 and S_N, then compare.

    slab1g examples/fuel_moderator.toml --sn 4 8 16 --plot flux.png --csv flux.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .problem import SlabProblem
from .solvers import Result, solve_all

# Categorical slots 1-2 of the validated default palette; transport (the
# reference) is drawn in primary ink. Line style is a secondary encoding.
STYLE = {
    "transport": dict(color="#1a1a19", ls="-", lw=2.0),
    "diffusion": dict(color="#2a78d6", ls="--", lw=2.0),
    "SP3": dict(color="#eb6834", ls="-.", lw=2.0),
}


def compare(results: list[Result]) -> str:
    ref = results[-1]
    lines = [f"Reference: {ref.method}", ""]
    hdr = f"{'method':<15s}{'k':>10s}{'dk [pcm]':>11s}{'max|dphi|/phi':>15s}{'rms dphi':>11s}"
    hdr += "".join(f"{'zone ' + str(i + 1):>11s}" for i in range(len(ref.zone_avg)))
    lines += [hdr, "-" * len(hdr)]
    for r in results:
        rel = (r.phi - ref.phi) / ref.phi
        dk = 1e5 * (r.k - ref.k) / ref.k
        row = f"{r.method:<15s}{r.k:>10.6f}{dk:>11.1f}{np.max(np.abs(rel)):>15.3e}{np.sqrt(np.mean(rel**2)):>11.3e}"
        row += "".join(f"{v:>11.5f}" for v in r.zone_avg)
        lines.append(row)
    lines += ["", "Fluxes normalised to slab-average scalar flux = 1."]
    return "\n".join(lines)


def write_csv(results: list[Result], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x"] + [r.method for r in results])
        for i, x in enumerate(results[0].x):
            w.writerow([f"{x:.8g}"] + [f"{r.phi[i]:.10g}" for r in results])


def plot(problem: SlabProblem, results: list[Result], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = results[-1]
    show = [r for r in results if r.method in ("diffusion", "SP3")] + [ref]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 2], hspace=0.08))
    for ax in (ax1, ax2):
        for i, (a, b) in enumerate(zip(problem.interfaces[:-1], problem.interfaces[1:])):
            if i % 2:
                ax.axvspan(a, b, color="#000000", alpha=0.04, lw=0)
        for xi in problem.interfaces[1:-1]:
            ax.axvline(xi, color="#8a8a85", lw=0.8)
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for r in show:
        st = STYLE["transport" if r is ref else r.method]
        lbl = f"{r.method}  (k = {r.k:.5f})"
        ax1.plot(r.x, r.phi, label=lbl, **st)
        if r is not ref:
            ax2.plot(r.x, 100 * (r.phi - ref.phi) / ref.phi, label=r.method, **st)
    ax2.axhline(0, color="#1a1a19", lw=0.8)

    ymax = ax1.get_ylim()[1]
    for z, a, b in zip(problem.zones, problem.interfaces[:-1], problem.interfaces[1:]):
        ax1.text(0.5 * (a + b), ymax, z.name, ha="center", va="top", color="#5f5e5a", fontsize=9)

    ax1.set_ylabel("scalar flux (slab average = 1)")
    ax1.set_title(f"{problem.title}: one-group flux, reflective boundaries", loc="left",
                  fontsize=11, pad=28)
    ax1.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
               ncol=3, borderaxespad=0.2, handlelength=2.6)
    ax2.set_ylabel(f"difference vs {ref.method.split()[0]} [%]")
    ax2.set_xlabel("x [cm]")
    ax2.set_xlim(0, problem.length)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, help="TOML problem file (see examples/)")
    p.add_argument("--sn", type=int, nargs="+", default=[16],
                   help="even S_N orders; the last one is the transport reference (default 16)")
    p.add_argument("--plot", type=Path, help="save flux comparison figure (png/pdf/svg)")
    p.add_argument("--csv", type=Path, help="save sampled fluxes as CSV")
    a = p.parse_args(argv)

    problem = SlabProblem.from_toml(a.config)
    print(f"{problem.title}  |  L = {problem.length:g} cm, {len(problem.zones)} zones")
    for i, z in enumerate(problem.zones, 1):
        print(f"  zone {i} {z.name:<10s} t={z.thickness:g}  St={z.sigma_t:.4g}  Ss={z.sigma_s:.4g}  "
              f"Sc={z.sigma_c:.4g}  Sf={z.sigma_f:.4g}  nu={z.nu:g}  ({z.thickness * z.sigma_t:.2f} mfp)")
        if z.sigma_s_l:
            mom = ", ".join(f"Ss{l}={v:.4g}" for l, v in enumerate(z.sigma_s_l, 1))
            print(f"         anisotropic P{z.scattering_order}: {mom}  (mean cosine {z.sigma_s_l[0] / z.sigma_s:.3f})")
    print()
    results = solve_all(problem, sorted(a.sn))
    print(compare(results))
    if a.plot:
        plot(problem, results, a.plot)
        print(f"\nfigure -> {a.plot}")
    if a.csv:
        write_csv(results, a.csv)
        print(f"csv    -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
