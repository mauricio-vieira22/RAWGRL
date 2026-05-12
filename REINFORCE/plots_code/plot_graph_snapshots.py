from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plots_code.common import ensure_dir, build_dummy_distributions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="plots/graphs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arrival_rate", type=float, default=2.0)
    p.add_argument("--mean_duration", type=float, default=10.0)
    p.add_argument("--total_timesteps", type=int, default=50)
    p.add_argument("--decision_period", type=int, default=5)
    p.add_argument("--sticky_mode", default="sticky", choices=["full", "sticky", "lite"])
    p.add_argument("--n_aps_dummy", type=int, default=6)
    p.add_argument("--joblib", default=None)
    p.add_argument("--every_tau", type=int, default=1)
    p.add_argument("--max_snapshots", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(
        Path(args.outdir) / f"sticky-{args.sticky_mode}_T-{args.decision_period}_seed-{args.seed}"
    )

    from model.network_graph_env import NetworkGraphEnv

    if args.joblib is None:
        dists = build_dummy_distributions(args.n_aps_dummy)
        n_aps = args.n_aps_dummy
    else:
        import joblib

        dists = joblib.load(args.joblib)
        n_aps = len(dists[0].blocks[0].datos)

    env = NetworkGraphEnv(
        distributions=dists,
        n_aps=n_aps,
        arrival_rate=args.arrival_rate,
        mean_duration=args.mean_duration,
        total_timesteps=args.total_timesteps,
        decision_period=args.decision_period,
        sticky_mode=args.sticky_mode,
        random_seed=args.seed,
        device="cpu",
    )

    obs, _ = env.reset(seed=args.seed)

    import matplotlib.pyplot as plt

    def draw_graph(hetero, fname: Path, title: str):
        n_ap = hetero["ap"].x.size(0) if "ap" in hetero.node_types else 0
        n_cl = hetero["client"].x.size(0) if "client" in hetero.node_types else 0

        ap_x = np.linspace(0, 1, max(n_ap, 1))
        cl_x = np.linspace(0, 1, max(n_cl, 1))
        ap_pos = {i: (ap_x[i], 1.0) for i in range(n_ap)}
        cl_pos = {i: (cl_x[i], 0.0) for i in range(n_cl)}

        plt.figure(figsize=(10, 4))
        if n_ap > 0:
            plt.scatter([ap_pos[i][0] for i in range(n_ap)], [1.0] * n_ap, s=120, c="tab:blue", label="AP")
        if n_cl > 0:
            plt.scatter([cl_pos[i][0] for i in range(n_cl)], [0.0] * n_cl, s=40, c="tab:green", label="Client")

        et = ("ap", "connects", "client")
        if et in hetero.edge_types:
            ei = hetero[et].edge_index
            if ei.numel() > 0:
                src = ei[0].cpu().numpy()
                dst = ei[1].cpu().numpy()
                for s, d in zip(src, dst):
                    x0, y0 = ap_pos[int(s)]
                    x1, y1 = cl_pos[int(d)]
                    plt.plot([x0, x1], [y0, y1], color="0.75", lw=0.7, alpha=0.7)

        et2 = ("client", "connected_to", "ap")
        if et2 in hetero.edge_types:
            ei = hetero[et2].edge_index
            if ei.numel() > 0:
                src = ei[0].cpu().numpy()
                dst = ei[1].cpu().numpy()
                for s, d in zip(src, dst):
                    x0, y0 = cl_pos[int(s)]
                    x1, y1 = ap_pos[int(d)]
                    plt.plot([x0, x1], [y0, y1], color="tab:red", lw=1.2, alpha=0.9)

        plt.axis("off")
        plt.title(title)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(fname, dpi=160)
        plt.close()

    done = False
    tau = 0
    saved = 0
    while not done and saved < args.max_snapshots:
        if tau % args.every_tau == 0:
            draw_graph(obs, outdir / f"graph_tau_{tau:04d}.png", title=f"__v0__ τ={tau}")
            saved += 1

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        tau += 1

    print(f"[OK] Graph snapshots en `{outdir}` ({saved} archivos)")


if __name__ == "__main__":
    main()

