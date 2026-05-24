from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RolloutConfig:
    seed: int
    arrival_rate: float
    mean_duration: float
    total_timesteps: int
    decision_period: int
    sticky_mode: str  # "full" | "sticky" | "lite"
    policy: str = "random"
    n_aps_dummy: int = 6
    joblib_path: str | None = None


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_dummy_distributions(n_aps: int = 6):
    import numpy as np
    import pandas as pd

    from REINFORCE.data.clases import Distribution, Block

    rows = []
    for ap in range(n_aps):
        rows.append({"ap": f"ap{ap}", "G_2_4": -60.0 - 2.0 * ap, "G_5": -65.0 - 2.0 * ap})
    df = pd.DataFrame(rows)
    blk = Block(block_idx=0, distribution_idx=0, datos=df)
    dist = Distribution(distribution_idx=0, mac_client="dummy", blocks=np.array([blk], dtype=object))
    return [dist]


def count_handovers(asignaciones_np: np.ndarray) -> int:
    ap = asignaciones_np[:, :, 0]
    valid = ~np.isnan(ap)
    a0 = ap[:, :-1]
    a1 = ap[:, 1:]
    v0 = valid[:, :-1]
    v1 = valid[:, 1:]
    changes = (a0 != a1) & v0 & v1
    return int(changes.sum())


def rollout(cfg: RolloutConfig) -> tuple[pd.DataFrame, dict]:
    from model.network_graph_env import NetworkGraphEnv

    if cfg.joblib_path is None:
        dists = build_dummy_distributions(cfg.n_aps_dummy)
        n_aps = cfg.n_aps_dummy
    else:
        import joblib

        dists = joblib.load(cfg.joblib_path)
        n_aps = len(dists[0].blocks[0].datos)

    env = NetworkGraphEnv(
        distributions=dists,
        n_aps=n_aps,
        arrival_rate=cfg.arrival_rate,
        mean_duration=cfg.mean_duration,
        total_timesteps=cfg.total_timesteps,
        decision_period=cfg.decision_period,
        sticky_mode=cfg.sticky_mode,
        random_seed=cfg.seed,
        device="cpu",
    )

    obs, _ = env.reset(seed=cfg.seed)
    rows: list[dict] = []
    done = False
    tau = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        rows.append(
            {
                "tau": tau,
                "reward_R_tau": float(reward),
                "n_active_clients": int(info.get("n_active_clients", 0)),
                "delta_t": float(info.get("delta_t", np.nan)),
                "epsilon_t_mean": float(info.get("epsilon_t_mean", np.nan)),
                "timestep_start": int(info.get("timestep_start", -1)),
                "timestep_end": int(info.get("timestep_end", -1)),
            }
        )
        tau += 1

    handovers = None
    asign = getattr(env, "_asignaciones", None)
    if asign is not None:
        try:
            handovers = count_handovers(asign.detach().cpu().numpy())
        except Exception:
            handovers = None

    meta = {
        "version": "__v0__",
        "seed": cfg.seed,
        "arrival_rate": cfg.arrival_rate,
        "mean_duration": cfg.mean_duration,
        "total_timesteps": cfg.total_timesteps,
        "decision_period": cfg.decision_period,
        "sticky_mode": cfg.sticky_mode,
        "handovers": handovers,
        "joblib": cfg.joblib_path,
    }

    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    return df, meta

