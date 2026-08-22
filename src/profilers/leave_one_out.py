"""Leave-one-out sensitivity profile for the all-analog deployment.

Phase 1 measures each projection *in isolation* (one projection analog, the
other 48 digital).  Deployment noises all 49 projections simultaneously, so the
additive profile is an assumption.  This module measures the complementary
leave-one-out (LOO) profile:

    s_loo(p) = E_a[ NLL(all 49 analog, noisy)  -  NLL(all analog, noisy, p restored) ]

with the *same* per-projection standard-normal fields Z_{p,a} (same seeds as
Phase 1: ``projection_noise_seed(realization_seed, projection_id)``), the same
reference noise level, and the same antithetic pairs, so the two profiles are
exactly paired.  ``restore_mode``:

* ``nominal`` (default): p stays analog (clipped + quantized) but noise-free,
  which isolates p's *noise* contribution exactly as Eq. (1) of the paper does.
* ``digital``: p is swapped back to its original floating-point module.

The profile is consumed only as a rank-stability check; see
``experiments/phase1_sensitivity/run_leave_one_out.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from src.common.analog import (
    ManualAnalogSettings,
    materialize_manual_noise,
    projection_noise_seed,
    set_analog_weights_exact,
    set_seed,
)
from src.common.metrics import evaluate_nll_ppl, summarize
from src.evaluation.aihwkit_gpt2 import HybridAnalogModel

Batch = Mapping[str, Tensor]


def rank_agreement(reference: Sequence[float], candidate: Sequence[float],
                   top_k: Iterable[int] = (5, 10, 20)) -> dict[str, float]:
    """Spearman rho, Kendall tau, and top-k overlap between two score vectors."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape or ref.ndim != 1 or ref.size < 3:
        raise ValueError("rank_agreement expects two equal-length 1-D vectors (n >= 3).")
    n = ref.size

    def _ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(-x, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        return ranks

    r1, r2 = _ranks(ref), _ranks(cand)
    rho = float(1.0 - 6.0 * np.sum((r1 - r2) ** 2) / (n * (n * n - 1)))
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = np.sign(ref[i] - ref[j])
            b = np.sign(cand[i] - cand[j])
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    tau = float((concordant - discordant) / (n * (n - 1) / 2))
    out = {"spearman_rho": rho, "kendall_tau": tau, "n": float(n)}
    ref_order = list(np.argsort(-ref, kind="mergesort"))
    cand_order = list(np.argsort(-cand, kind="mergesort"))
    for k in top_k:
        k = int(min(k, n))
        out[f"top{k}_overlap"] = float(len(set(ref_order[:k]) & set(cand_order[:k])) / k)
    return out


class LeaveOneOutProfiler:
    """All-49-analog counterpart of :class:`AIHWKITSensitivityProfiler`."""

    def __init__(
        self,
        model: Any,
        config: Mapping[str, Any],
        phase1_rows: Sequence[Mapping[str, Any]],
        *,
        restore_mode: str = "nominal",
    ) -> None:
        if restore_mode not in {"nominal", "digital"}:
            raise ValueError("restore_mode must be 'nominal' or 'digital'.")
        self.model = model
        self.config = config
        self.device = torch.device(str(config["model"]["device"]))
        self.settings = ManualAnalogSettings.from_config(config)
        self.settings.validate()
        profile = config["profiling"]
        self.num_seeds = int(profile["num_seeds"])
        self.seed_stride = int(profile.get("seed_stride", 1))
        self.antithetic = bool(profile.get("antithetic", True))
        self.include_lm_head = bool(profile.get("include_lm_head", True))
        self.base_seed = int(config["experiment"]["seed"])
        if self.num_seeds <= 0 or self.seed_stride <= 0:
            raise ValueError("num_seeds and seed_stride must be positive.")
        self.restore_mode = restore_mode
        self.phase1_rows = list(phase1_rows)
        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()
        # Every Phase-1 projection becomes analog; nothing is protected.
        self.hybrid = HybridAnalogModel(
            self.model,
            digital_projection_ids=[],
            settings=self.settings,
            include_lm_head_candidate=self.include_lm_head,
            phase1_projection_rows=self.phase1_rows,
        ).convert()

    # ------------------------------------------------------------------ noise
    def _field(self, projection_id: str, realization: int) -> tuple[Tensor, int]:
        """Same standard-normal field and seed as Phase 1 for this projection."""
        realization_seed = self.base_seed + realization * self.seed_stride
        seed = projection_noise_seed(realization_seed, projection_id)
        set_seed(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        shape = self.hybrid.states[projection_id].clipped_weight.shape
        return torch.randn(tuple(shape), generator=generator, dtype=torch.float32), seed

    def _write_noisy(self, projection_id: str, realization: int, sign: float) -> None:
        state = self.hybrid.states[projection_id]
        z, _ = self._field(projection_id, realization)
        noisy, _ = materialize_manual_noise(
            state.clipped_weight.detach().cpu(),
            self.settings.reference_noise_std,
            state.programmed_range,
            z * sign,
        )
        set_analog_weights_exact(state.analog_module, noisy, state.bias, verify=False)

    def _write_nominal(self, projection_id: str) -> None:
        state = self.hybrid.states[projection_id]
        set_analog_weights_exact(state.analog_module, state.clipped_weight, state.bias, verify=False)

    def _restore_one(self, projection_id: str) -> None:
        """Remove projection p's noise according to ``restore_mode``."""
        if self.restore_mode == "nominal":
            self._write_nominal(projection_id)
        else:
            state = self.hybrid.states[projection_id]
            setattr(state.handle.parent, state.handle.attribute,
                    self.hybrid.original_modules[projection_id])

    def _unrestore_one(self, projection_id: str, realization: int, sign: float) -> None:
        if self.restore_mode == "digital":
            state = self.hybrid.states[projection_id]
            setattr(state.handle.parent, state.handle.attribute, state.analog_module)
        self._write_noisy(projection_id, realization, sign)

    # ------------------------------------------------------------ checkpoint
    @staticmethod
    def _key(realization: int, sign: float, projection_id: str | None = None) -> str:
        base = f"{int(realization)}|{sign:+.0f}"
        return base if projection_id is None else f"{base}|{projection_id}"

    @staticmethod
    def _load_checkpoint(path: Path, signature: Mapping[str, Any] | None,
                         log: bool) -> dict[str, Any]:
        """Return the cached evaluations at ``path`` if its signature matches.

        A checkpoint written under a different signature (other restore mode,
        batch count, noise level, Phase-1 artifact, ...) must never be reused:
        it is moved aside with a ``_stale_<stamp>`` suffix and a fresh run starts.
        """
        empty = {"nominal": None, "all_nll": {}, "loo_nll": {}}
        if not path.is_file():
            return empty
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if log:
                print(f"[LOO] checkpoint {path} unreadable ({exc}); starting fresh.", flush=True)
            return empty
        stored = payload.get("signature")
        if signature is not None and stored != dict(signature):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            stale = path.with_name(f"{path.stem}_stale_{stamp}{path.suffix}")
            os.replace(path, stale)
            if log:
                print(f"[LOO] checkpoint signature differs from this run; moved it to {stale} "
                      f"and starting fresh.", flush=True)
            return empty
        cached = {
            "nominal": payload.get("nominal"),
            "all_nll": dict(payload.get("all_nll", {})),
            "loo_nll": dict(payload.get("loo_nll", {})),
        }
        if log:
            print(f"[LOO] resuming from {path}: {len(cached['all_nll'])} all-analog and "
                  f"{len(cached['loo_nll'])} leave-one-out evaluations cached.", flush=True)
        return cached

    @staticmethod
    def _save_checkpoint(path: Path, signature: Mapping[str, Any] | None,
                         cached: Mapping[str, Any]) -> None:
        payload = {
            "signature": None if signature is None else dict(signature),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "nominal": cached["nominal"],
            "all_nll": cached["all_nll"],
            "loo_nll": cached["loo_nll"],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------ main
    def run(self, batches: list[Batch], *, projection_ids: Sequence[str] | None = None,
            log: bool = True, checkpoint_path: Path | None = None,
            checkpoint_signature: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Measure the LOO profile.

        With ``checkpoint_path`` every forward-pass result is persisted as soon as
        it is measured, and a later call with the same ``checkpoint_signature``
        skips everything already cached -- so a run interrupted by a runtime
        disconnect resumes where it stopped instead of starting over.
        """
        ids = list(self.hybrid.analog_projection_ids)
        if projection_ids is None:
            targets = ids
        else:
            unknown = sorted(set(projection_ids) - set(ids))
            if unknown:
                raise ValueError(f"Unknown projection ids for leave-one-out: {unknown}")
            targets = [p for p in ids if p in set(projection_ids)]
            if not targets:
                raise ValueError("No projections selected for leave-one-out.")
        phase1_by_id = {str(r["projection_id"]): r for r in self.phase1_rows}
        signs = (1.0, -1.0) if self.antithetic else (1.0,)

        cached = (self._load_checkpoint(checkpoint_path, checkpoint_signature, log)
                  if checkpoint_path is not None else
                  {"nominal": None, "all_nll": {}, "loo_nll": {}})

        def persist() -> None:
            if checkpoint_path is not None:
                self._save_checkpoint(checkpoint_path, checkpoint_signature, cached)

        # zero-noise all-analog reference (clip + quantization only)
        if cached["nominal"] is None:
            self.hybrid.restore_nominal_weights()
            nominal_nll, nominal_ppl, token_count = evaluate_nll_ppl(self.model, batches, self.device)
            cached["nominal"] = {"nll": float(nominal_nll), "ppl": float(nominal_ppl),
                                 "tokens": int(token_count)}
            persist()
        nominal_nll = float(cached["nominal"]["nll"])
        nominal_ppl = float(cached["nominal"]["ppl"])
        token_count = int(cached["nominal"]["tokens"])

        all_nll: dict[tuple[int, float], float] = {}
        loo_nll: dict[str, dict[tuple[int, float], float]] = {p: {} for p in targets}
        # progress bookkeeping: passes measured in this process and their wall time
        total_passes = self.num_seeds * len(signs) * (1 + len(targets))
        done_passes = len(cached["all_nll"]) + len(cached["loo_nll"])
        passes_this_run = 0
        seconds_this_run = 0.0

        def progress() -> str:
            if passes_this_run == 0:
                return ""
            per_pass = seconds_this_run / passes_this_run
            remaining = max(total_passes - done_passes, 0) * per_pass
            return f"  [{done_passes}/{total_passes} passes, {per_pass:.0f} s/pass, ETA {remaining / 3600:.1f} h]"

        try:
            for realization in range(self.num_seeds):
                for sign in signs:
                    key_all = self._key(realization, sign)
                    pending = [p for p in targets if self._key(realization, sign, p) not in cached["loo_nll"]]
                    if key_all in cached["all_nll"] and not pending:
                        all_nll[(realization, sign)] = float(cached["all_nll"][key_all])
                        for p in targets:
                            loo_nll[p][(realization, sign)] = float(cached["loo_nll"][self._key(realization, sign, p)])
                        continue
                    for p in ids:
                        self._write_noisy(p, realization, sign)
                    if key_all in cached["all_nll"]:
                        nll_all = float(cached["all_nll"][key_all])
                    else:
                        t0 = time.monotonic()
                        nll_all, _, _ = evaluate_nll_ppl(self.model, batches, self.device)
                        seconds_this_run += time.monotonic() - t0
                        passes_this_run += 1
                        done_passes += 1
                        cached["all_nll"][key_all] = float(nll_all)
                        persist()
                    all_nll[(realization, sign)] = nll_all
                    if log:
                        print(f"[LOO] realization {realization} sign {sign:+.0f}: all-analog NLL {nll_all:.5f}{progress()}", flush=True)
                    for index, p in enumerate(targets, start=1):
                        key_p = self._key(realization, sign, p)
                        if key_p in cached["loo_nll"]:
                            loo_nll[p][(realization, sign)] = float(cached["loo_nll"][key_p])
                            continue
                        self._restore_one(p)
                        t0 = time.monotonic()
                        try:
                            nll_minus, _, _ = evaluate_nll_ppl(self.model, batches, self.device)
                        finally:
                            # Re-attach/re-noise p even if evaluation fails, so the
                            # model never stays in a partially-digital state.
                            self._unrestore_one(p, realization, sign)
                        seconds_this_run += time.monotonic() - t0
                        passes_this_run += 1
                        done_passes += 1
                        loo_nll[p][(realization, sign)] = nll_minus
                        cached["loo_nll"][key_p] = float(nll_minus)
                        persist()
                        if log:
                            print(f"[LOO]   {index}/{len(targets)} {p}: delta {nll_all - nll_minus:+.5f}{progress()}", flush=True)
        finally:
            self.hybrid.restore_nominal_weights()
            self.hybrid.assert_nominal_restored()

        rows: list[dict[str, Any]] = []
        for p in targets:
            state = self.hybrid.states[p]
            per_real = []
            for realization in range(self.num_seeds):
                deltas = [all_nll[(realization, s)] - loo_nll[p][(realization, s)] for s in signs]
                per_real.append({
                    "realization": realization,
                    "delta_nll_loo": float(np.mean(deltas)),
                    "nll_all_analog": float(np.mean([all_nll[(realization, s)] for s in signs])),
                    "nll_without_p": float(np.mean([loo_nll[p][(realization, s)] for s in signs])),
                })
            stats = summarize([r["delta_nll_loo"] for r in per_real])
            iso = phase1_by_id.get(p, {})
            rows.append({
                "projection_id": p,
                "role": state.handle.role,
                "block_index": state.handle.block_index,
                "sensitivity_isolated": float(iso.get("sensitivity_score_for_mapping", float("nan"))),
                "sensitivity_loo_mean": float(stats["mean"]),
                "sensitivity_loo_std": float(stats["std"]),
                "sensitivity_loo_sem": float(stats["sem"]),
                "realizations": per_real,
            })

        iso_scores = [r["sensitivity_isolated"] for r in rows]
        loo_scores = [r["sensitivity_loo_mean"] for r in rows]
        agreement = rank_agreement(iso_scores, loo_scores) if len(rows) >= 3 else {}
        sum_iso = float(np.nansum(iso_scores))
        sum_loo = float(np.sum(loo_scores))
        mean_all = float(np.mean(list(all_nll.values())))
        return {
            "restore_mode": self.restore_mode,
            "analog_projection_count": len(ids),
            "profiled_projection_count": len(targets),
            "nominal_all_analog_nll": nominal_nll,
            "nominal_all_analog_ppl": nominal_ppl,
            "noisy_all_analog_nll_mean": mean_all,
            "noisy_all_analog_ppl_mean": math.exp(mean_all),
            "predicted_tokens": token_count,
            "sum_isolated_sensitivity": sum_iso,
            "sum_loo_sensitivity": sum_loo,
            "total_noise_penalty_all_analog": mean_all - nominal_nll,
            "rank_agreement": agreement,
            "projections": rows,
        }
