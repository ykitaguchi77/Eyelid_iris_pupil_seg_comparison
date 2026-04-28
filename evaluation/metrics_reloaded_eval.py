"""MetricsReloaded wrapper for 2D binary segmentation evaluation.

Computes DSC, HD95, NSD per structure (eyelid, iris, pupil) using the
official reference implementation by Maier-Hein et al., Nat Methods 2024.

Reference:
    Maier-Hein L, Reinke A, et al. Metrics reloaded: recommendations for
    image analysis validation. Nat Methods 2024;21:195-212.
"""
from __future__ import annotations

import contextlib
import io
import warnings
from dataclasses import dataclass

import numpy as np
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures

# MetricsReloaded 0.1.0 has leftover debug print statements in
# BinaryPairwiseMeasures.__init__ -> calculate_worse_dist (line 307, 309 of
# pairwise_measures.py). With ~90k metric calls over a full 5-fold CV, these
# prints flood Jupyter IOPub and throttle the kernel to 10-100x slowdown.
# _silence redirects stdout during BPM construction to suppress them.
_DEV_NULL = io.StringIO()


@contextlib.contextmanager
def _silence():
    with warnings.catch_warnings(), contextlib.redirect_stdout(_DEV_NULL):
        warnings.simplefilter("ignore")
        _DEV_NULL.seek(0)
        _DEV_NULL.truncate(0)
        yield

# NSD tolerance (pixels) per anatomical structure.
# Rationale: larger for big & well-defined structures, smaller for small
# structures where sub-pixel precision matters.
DEFAULT_NSD_TAU = {"eyelid": 2.0, "iris": 2.0, "pupil": 1.0}


@dataclass
class StructureMetrics:
    dsc: float
    hd95: float
    nsd: float
    nsd_tau: float


def eval_binary_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    nsd_tau: float = 1.0,
    hd_percentile: int = 95,
) -> StructureMetrics:
    """Evaluate a single 2D binary mask pair.

    Args:
        pred: (H, W) binary mask, 0/1 or 0/255 (uint8 or bool).
        gt:   (H, W) binary mask, same convention.
        nsd_tau: NSD tolerance in pixels.
        hd_percentile: percentile for HD (95 by default).

    Returns:
        StructureMetrics with DSC, HD95, NSD. NaN if both masks empty.
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)

    if p.sum() == 0 and g.sum() == 0:
        return StructureMetrics(dsc=np.nan, hd95=np.nan, nsd=np.nan, nsd_tau=nsd_tau)

    with _silence():
        bpm = BinaryPairwiseMeasures(
            p,
            g,
            measures=["dsc", "hd_perc", "nsd"],
            dict_args={"hd_perc": hd_percentile, "nsd": nsd_tau},
        )
        res = bpm.to_dict_meas()

    return StructureMetrics(
        dsc=float(res["dsc"]),
        hd95=float(res["hd_perc"]),
        nsd=float(res["nsd"]),
        nsd_tau=nsd_tau,
    )


def compute_hd95_nsd(pred: np.ndarray, gt: np.ndarray, tau: float) -> tuple[float, float]:
    """Return (HD95, NSD) only, without re-computing DSC.

    Use this when DSC is already computed elsewhere with a project-specific
    formula and only boundary metrics are needed.
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return float("nan"), float("nan")
    with _silence():
        bpm = BinaryPairwiseMeasures(
            p, g,
            measures=["hd_perc", "nsd"],
            dict_args={"hd_perc": 95, "nsd": tau},
        )
        res = bpm.to_dict_meas()
    return float(res["hd_perc"]), float(res["nsd"])


def eval_three_structures(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[str, np.ndarray],
    tau_per_structure: dict[str, float] | None = None,
) -> dict[str, StructureMetrics]:
    """Evaluate eyelid / iris / pupil triplet.

    Args:
        pred_masks: dict with keys {"eyelid", "iris", "pupil"} -> (H, W) binary.
        gt_masks:   dict with same keys and shapes.
        tau_per_structure: NSD tau override per structure; defaults to DEFAULT_NSD_TAU.

    Returns:
        dict with same keys -> StructureMetrics.
    """
    tau = tau_per_structure or DEFAULT_NSD_TAU
    out = {}
    for name in ("eyelid", "iris", "pupil"):
        out[name] = eval_binary_pair(
            pred_masks[name], gt_masks[name], nsd_tau=tau[name]
        )
    return out


if __name__ == "__main__":
    # Smoke test with a synthetic 2D pair.
    H, W = 512, 512
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    gt = ((yy - 256) ** 2 + (xx - 256) ** 2 < 80 ** 2).astype(np.uint8)
    pred = ((yy - 258) ** 2 + (xx - 256) ** 2 < 78 ** 2).astype(np.uint8)
    m = eval_binary_pair(pred, gt, nsd_tau=2.0)
    print(f"Synthetic circle pair: DSC={m.dsc:.4f}, HD95={m.hd95:.2f}, NSD(tau=2)={m.nsd:.4f}")
