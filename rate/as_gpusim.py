"""GpuSim facade over a RateSim, so paper 0's protocol code runs on the rate model unchanged
(paper 1, PRD phase 1). learn/condition.py and learn/persist.py only call run_batch and
set_plastic; this answers both with GpuSim's contracts:

  run_batch: drives are (idx, prob per DT step) as condition.to_prob makes them -> Hz =
             prob * 1000 / G.DT. Returns rates * t_run / 1000 as float COUNTS [B, N]; callers
             divide by seconds, so they recover the rates up to float rounding.
  set_plastic: values in mV (x W_SYN, as Plastic.push writes them) -> synapse counts.

Seeds are ignored: the rate engine is deterministic (plan D2). State is RateSim's v [N, B].
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpu_sim as G  # noqa: E402

_C = np.float32(G.W_SYN)  # the constant Plastic.signed_values multiplies by


def mv_to_syn(values):
    """Invert Plastic.signed_values' float32 x W_SYN. Plain division misses by 1 ulp on 8 of the
    18,674 baseline edges; where an integer count maps to exactly this float32 value, return it."""
    v = np.asarray(values, np.float32)
    syn = v.astype(np.float64) / np.float64(_C)
    r = np.rint(syn)
    return np.where((r * _C).astype(np.float32) == v, r, syn)


class RateAsGpuSim:
    def __init__(self, rsim):
        self.r = rsim

    def run_batch(self, drives, seeds, t_run=300.0, state=None, return_state=False, **kw):
        assert all(v in (None, "counter") for v in kw.values()), "unsupported run_batch args %s" % kw
        d = [(np.asarray(i, np.int64),
              torch.as_tensor(np.asarray(p, np.float64) * 1000.0 / G.DT, dtype=self.r.dtype, device=self.r.device))
             for i, p in drives]
        with torch.no_grad():
            out = self.r.run(d, t_run, state=state, return_state=return_state)
        rates, st = out if return_state else (out, None)
        counts = rates * (t_run / 1000.0)
        return (counts, st) if return_state else counts

    def set_plastic(self, offsets, values):
        self.r.set_plastic(offsets, mv_to_syn(values))
