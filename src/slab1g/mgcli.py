"""Command-line output for multigroup problems (called from cli.main)."""
from __future__ import annotations

import csv

import numpy as np

from .cli import NORM_TEXT, STYLE
from .multigroup import MGProblem, MGResult, solve_all

YLABEL = {"average": "scalar flux (slab- and group-summed average = 1)",
          "fission": "scalar flux (total fission rate = 1)"}


def describe(problem: MGProblem) -> str:
    G = problem.G
    lines = [f"{problem.title}  |  L = {problem.length:g} cm, {len(problem.zones)} zones, {G} groups"]
    for i, z in enumerate(problem.zones, 1):
        D = np.linalg.inv(z.R(1)) / 3.0
        lines.append(f"  zone {i} {z.name:<10s} t={z.thickness:g}  chi={np.round(z.chi, 4).tolist()}")
        for g in range(G):
            lines.append(f"     g{g + 1}: St={z.sigma_t_g[g]:.4g}  Sa={z.sigma_a_g[g]:.4g}  nuSf={z.nu_sigma_f[g]:.4g}  "
                         f"Ss0(g->*)={np.round(z.S(0)[g], 5).tolist()}  D={D[g, g]:.4g}  "
                         f"({z.thickness * z.sigma_t_g[g]:.2f} mfp)")
        if z.L:
            lines.append(f"     anisotropic up to P{z.L}; Ss1 = {np.round(z.S(1), 5).tolist()}")
    return "\n".join(lines)


def compare(results: list[MGResult], problem: MGProblem, norm: str) -> str:
    ref = results[-1]
    G, Z = problem.G, len(problem.zones)
    lines = [f"Reference: {ref.method}", ""]
    hdr = f"{'method':<16s}{'k':>10s}{'dk [pcm]':>10s}" + "".join(f"{'max|dphi|/phi g' + str(g + 1):>18s}" for g in range(G))
    hdr += "".join(f"{f'g{g + 1} zone {i + 1}':>12s}" for g in range(G) for i in range(Z))
    lines += [hdr, "-" * len(hdr)]
    for r in results:
        row = f"{r.method:<16s}{r.k:>10.6f}{1e5 * (r.k - ref.k) / ref.k:>10.1f}"
        row += "".join(f"{np.max(np.abs(r.phi[g] / ref.phi[g] - 1)):>18.3e}" for g in range(G))
        row += "".join(f"{r.zone_avg[g, i]:>12.5g}" for g in range(G) for i in range(Z))
        lines.append(row)
    lines += ["", NORM_TEXT[norm].replace("scalar flux", "scalar flux summed over groups")]
    return "\n".join(lines)


def write_csv(results: list[MGResult], path):
    G = results[0].phi.shape[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x"] + [f"{r.method} g{g + 1}" for r in results for g in range(G)])
        for i, x in enumerate(results[0].x):
            w.writerow([f"{x:.8g}"] + [f"{r.phi[g, i]:.10g}" for r in results for g in range(G)])


def plot(problem: MGProblem, results: list[MGResult], path, norm: str = "average"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = results[-1]
    show = [r for r in results if r.method.startswith(("P1", "SP3"))] + [ref]
    G = problem.G
    fig, axes = plt.subplots(G, 1, figsize=(8, 3.6 * G), sharex=True, squeeze=False,
                             gridspec_kw=dict(hspace=0.28))
    for g, ax in enumerate(axes[:, 0]):
        for i, (a, b) in enumerate(zip(problem.interfaces[:-1], problem.interfaces[1:])):
            if i % 2:
                ax.axvspan(a, b, color="#000000", alpha=0.04, lw=0)
        for xi in problem.interfaces[1:-1]:
            ax.axvline(xi, color="#8a8a85", lw=0.8)
        for r in show:
            key = "transport" if r is ref else ("diffusion" if r.method.startswith("P1") else "SP3")
            lbl = {"diffusion": "P1 (= diffusion)", "SP3": "SP3"}.get(key, ref.method) + f"  (k = {r.k:.5f})"
            ax.plot(r.x, r.phi[g], label=lbl if g == 0 else None, **STYLE[key])
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(problem.group_names[g], loc="left", fontsize=10, color="#3d3d3a")
        ax.set_ylabel("scalar flux")
        ymax = ax.get_ylim()[1]
        for z, a, b in zip(problem.zones, problem.interfaces[:-1], problem.interfaces[1:]):
            ax.text(0.5 * (a + b), ymax, z.name, ha="center", va="top", color="#5f5e5a", fontsize=9)
    axes[-1, 0].set_xlabel("x [cm]")
    axes[-1, 0].set_xlim(0, problem.length)
    fig.suptitle(f"{problem.title}: flux per group, reflective boundaries\n{YLABEL[norm]}",
                 x=0.125, y=1.04, ha="left", fontsize=11)
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.985), ncol=3, frameon=False, fontsize=9, handlelength=2.6)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(a) -> int:
    problem = MGProblem.from_toml(a.config)
    print(describe(problem))
    print()
    results = solve_all(problem, sorted(a.sn), a.norm)
    print(compare(results, problem, a.norm))
    if a.plot:
        plot(problem, results, a.plot, a.norm)
        print(f"\nfigure -> {a.plot}")
    if a.csv:
        write_csv(results, a.csv)
        print(f"csv    -> {a.csv}")
    return 0
