from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_code.common import RolloutConfig, ensure_dir, rollout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="plots/data")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sticky_modes", nargs="+", default=["full", "sticky", "lite"])
    p.add_argument("--decision_periods", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--arrival_rate", type=float, default=2.0)
    p.add_argument("--mean_duration", type=float, default=10.0)
    p.add_argument("--total_timesteps", type=int, default=100)
    p.add_argument("--joblib", default=None)
    p.add_argument("--n_aps_dummy", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(Path(args.outdir))
    all_rows = []

    for seed in args.seeds:
        for sm in args.sticky_modes:
            for T in args.decision_periods:
                cfg = RolloutConfig(
                    seed=seed,
                    arrival_rate=args.arrival_rate,
                    mean_duration=args.mean_duration,
                    total_timesteps=args.total_timesteps,
                    decision_period=T,
                    sticky_mode=sm,
                    joblib_path=args.joblib,
                    n_aps_dummy=args.n_aps_dummy,
                )
                df, meta = rollout(cfg)
                df.to_csv(outdir / f"rollout__REINFORCE__sticky-{sm}_T-{T}_seed-{seed}.csv", index=False)
                all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    big.to_csv(outdir / "rollouts_all.csv", index=False)
    print(f"[OK] Rollouts guardados en `{outdir}`")


if __name__ == "__main__":
    main()

