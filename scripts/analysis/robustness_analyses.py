#!/usr/bin/env python3
"""Map-error robustness and adaptive re-mapping envelope, both in the map-time proxy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common.config import load_yaml  # noqa: E402
from src.simulators.hardware import HardwareConfig  # noqa: E402
from src.simulators.tile_fidelity import TileFidelityModel  # noqa: E402
from src.mapping.sharding import build_shards  # noqa: E402
from src.mapping.placement import place_shards  # noqa: E402

CONFIG_PATH = REPO / "configs/full_pipeline/gpt2_allanalog_bounds.yaml"
PROFILE_PATH = REPO / "paper/data/phase1_sensitivity_seed42.json"
TRACE_SEEDS = [41, 42, 43, 44, 45]   # the paper's five hardware traces
EVAL_T = [0, 30, 60, 90, 119]        # the Phase-4 evaluation timesteps
TAUS = [0.05, 0.10, 0.20, 0.50]      # relative map-error levels
N_DRAWS = 200                        # measurement draws per (trace, tau)
MEASUREMENT_SEED = 20260827

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=CONFIG_PATH)
parser.add_argument(
    "--phase1",
    type=Path,
    default=PROFILE_PATH,
    help="Phase-1 profile JSON (the paper used the seed-42 profile, kept outside the repository).",
)
args = parser.parse_args()

config = load_yaml(args.config)
hardware = HardwareConfig.from_config(config)
hardware.validate()
profile = json.load(open(args.phase1))
tiers_per_tile = int(config["hardware"]["tiers_per_tile"])

shards = build_shards(
    profile["projections"],
    digital_projection_ids=[],
    tier_rows=int(config["hardware"]["tier_shape"]["rows"]),
    tier_cols=int(config["hardware"]["tier_shape"]["cols"]),
    sensitivity_floor=float(config["phase3"].get("sensitivity_floor", 0.0)),
)
assert len(shards) == 678, len(shards)
imp = np.array([s.importance for s in shards])
shard_order = {s.shard_id: k for k, s in enumerate(shards)}
traces = {s: TileFidelityModel(hardware, config["phase2"], s).generate() for s in TRACE_SEEDS}


def tile_ids(records) -> np.ndarray:
    out = np.empty(len(records), dtype=int)
    for r in records:
        out[shard_order[r.shard_id]] = r.tile_id
    return out


def phi(assign: np.ndarray, sigma_t: np.ndarray) -> float:
    return float(np.sum(imp * sigma_t[assign] ** 2))


def place(noise, available, policy):
    return place_shards(
        shards, noise=noise, available=available, tiers_per_tile=tiers_per_tile,
        policy=policy, timestep=0,
        seed=int(config["experiment"].get("placement_seed", 42)),
    )


# ---------------- M2: per-timestep Phi-optimal envelope ----------------
print("=== M2 (Sec. V-C): adaptive envelope vs one-shot placement ===")
rows = []
for s in TRACE_SEEDS:
    tr = traces[s]
    sig = tr.noise_std.astype(np.float64)
    traj_rms = np.sqrt(np.square(sig).mean(axis=0))
    a_static = tile_ids(place(sig[0], tr.available[0], "static_sensitivity"))
    a_oracle = tile_ids(place(traj_rms, tr.available.all(axis=0), "oracle_clairvoyant"))
    a_hw = tile_ids(place(sig[0], tr.available[0], "hardware_only"))
    m = {"static": [], "oracle": [], "hw": [], "adaptive": []}
    for t in EVAL_T:
        a_t = tile_ids(place(sig[t], tr.available[t], "static_sensitivity"))
        m["adaptive"].append(phi(a_t, sig[t]))
        for name, a in (("static", a_static), ("oracle", a_oracle), ("hw", a_hw)):
            m[name].append(phi(a, sig[t]))
    rows.append({k: float(np.mean(v)) for k, v in m.items()})

st = np.array([r["static"] for r in rows])
orc = np.array([r["oracle"] for r in rows])
ad = np.array([r["adaptive"] for r in rows])
hw = np.array([r["hw"] for r in rows])
print(f"lifetime Phi saving vs one-shot, adaptive envelope: {np.mean((st - ad) / st) * 100:.1f}%")
print(f"lifetime Phi saving vs one-shot, static clairvoyant: {np.mean((st - orc) / st) * 100:.1f}%")
print(f"adaptive headroom as share of the static-vs-hardware_only Phi gap: "
      f"{np.mean((st - ad) / (hw - st)) * 100:.1f}%")

# ---------------- M3: noisy tile-quality map ---------------------------
print("\n=== M3 (Sec. III-D): mapper given sigma_hat = sigma * exp(tau Z) ===")
rng = np.random.default_rng(MEASUREMENT_SEED)
for tau in TAUS:
    rets = []
    for s in TRACE_SEEDS:
        tr = traces[s]
        sig = tr.noise_std.astype(np.float64)
        a_ss = tile_ids(place(sig[0], tr.available[0], "static_sensitivity"))
        a_hw = tile_ids(place(sig[0], tr.available[0], "hardware_only"))
        exact_gap = np.mean([phi(a_hw, sig[t]) - phi(a_ss, sig[t]) for t in EVAL_T])
        for _ in range(N_DRAWS):
            sigma_hat = sig[0] * np.exp(tau * rng.standard_normal(sig[0].shape))
            b_ss = tile_ids(place(sigma_hat, tr.available[0], "static_sensitivity"))
            b_hw = tile_ids(place(sigma_hat, tr.available[0], "hardware_only"))
            g = np.mean([phi(b_hw, sig[t]) - phi(b_ss, sig[t]) for t in EVAL_T])
            rets.append(g / exact_gap)
    rets = np.array(rets)
    print(f"tau={tau:4.2f} (~{0.5 / tau**2:5.0f} write-verify samples): "
          f"margin retention mean={rets.mean() * 100:5.1f}%  "
          f"5th pct={np.percentile(rets, 5) * 100:5.1f}%")
