from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_code.common import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv", default="plots/data/rollouts_all.csv")
    p.add_argument("--outdir", default="plots/figures")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    df = pd.read_csv(args.data_csv)
    if df.empty:
        raise SystemExit("No hay datos. Corre primero `python plots_code/run_rollouts.py`.")

    import numpy as np
    import matplotlib.pyplot as plt

    grp = (
        df.groupby(["sticky_mode", "decision_period", "seed"], as_index=False)["reward_R_tau"]
        .mean()
        .rename(columns={"reward_R_tau": "mean_R_tau"})
    )

    plt.figure(figsize=(9, 3.5))
    labels, data = [], []
    for sm in ["full", "sticky", "lite"]:
        vals = grp[grp["sticky_mode"] == sm]["mean_R_tau"].values
        data.append(vals)
        labels.append(sm)
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.grid(True, alpha=0.3, axis="y")
    plt.ylabel("mean(R_τ) por episodio")
    plt.title("__v1__ — comparación sticky modes")
    plt.tight_layout()
    plt.savefig(outdir / "__v1__sticky_modes_boxplot.png", dpi=160)
    plt.close()

    agg = (
        grp.groupby(["sticky_mode", "decision_period"], as_index=False)["mean_R_tau"]
        .agg(["mean", "std"])
        .reset_index()
    )
    plt.figure(figsize=(9, 3.5))
    for sm in ["full", "sticky", "lite"]:
        ssm = agg[agg["sticky_mode"] == sm].sort_values("decision_period")
        if ssm.empty:
            continue
        x = ssm["decision_period"].values
        y = ssm["mean"].values
        e = ssm["std"].values
        plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=sm)
    plt.grid(True, alpha=0.3)
    plt.xlabel("decision_period (T)")
    plt.ylabel("mean(R_τ) por episodio")
    plt.title("__v1__ — efecto de decidir cada T slots")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "__v1__decision_period_effect.png", dpi=160)
    plt.close()

    sample = df.sample(min(len(df), 5000), random_state=0)
    plt.figure(figsize=(9, 3.5))
    plt.scatter(sample["delta_t"], sample["n_active_clients"], s=8, alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel("δ_t")
    plt.ylabel("U_t (clientes activos)")
    plt.title("__v1__ POMDP — δ_t vs carga (muestra)")
    plt.tight_layout()
    plt.savefig(outdir / "__v1__pomdp_delta_vs_load_scatter.png", dpi=160)
    plt.close()

    print(f"[OK] Figuras en `{outdir}`")


if __name__ == "__main__":
    main()

