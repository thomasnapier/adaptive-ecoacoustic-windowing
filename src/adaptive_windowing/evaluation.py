"""New implementation of the conditional criteria in paper v5, section 4.2.

These metrics supplement the original notebook metrics. They are geometry-tested,
not verified against the unavailable original Table 3 evaluation script or dataset.
"""

import numpy as np


def conditional_metrics(predictions, references):
    """Input: boxes in seconds/Hz. Process: frequency-filter + greedy matching. Output: metrics.

    Thresholded criteria use separate one-to-one matches. The primary ranking
    score for T-Gated is temporal IoMin (the paper does not explicitly name a
    composite score); ties use 2D IoMin, prediction purity, then IoU. Exact remaining
    ties are resolved by input order. Mean-best temporal IoMin permits reuse.
    """

    def overlap(a, b, k0, k1):
        return max(0.0, min(a[k1], b[k1]) - max(a[k0], b[k0]))

    kept = [
        p for p in predictions if any(overlap(p, g, "f0", "f1") > 0 for g in references)
    ]
    n, m = len(kept), len(references)
    if not m:
        return {
            "reference_count": 0,
            "raw_prediction_count": len(predictions),
            "evaluable_prediction_count": 0,
            "status": "empty_reference_set; conditional scores undefined",
        }
    pairs = []
    for i, p in enumerate(kept):
        pdur = p["t1"] - p["t0"]
        pbw = p["f1"] - p["f0"]
        pa = pdur * pbw
        for j, g in enumerate(references):
            gdur = g["t1"] - g["t0"]
            gbw = g["f1"] - g["f0"]
            ga = gdur * gbw
            ti = overlap(p, g, "t0", "t1")
            fi = overlap(p, g, "f0", "f1")
            area = ti * fi
            pairs.append(
                {
                    "i": i,
                    "j": j,
                    "iou": area / (pa + ga - area),
                    "iomin": area / min(pa, ga),
                    "temporal_iomin": ti / min(pdur, gdur),
                    "frequency_iomin": fi / min(pbw, gbw),
                    "purity": area / pa,
                }
            )

    def match(score, eligible):
        candidates = [r for r in pairs if eligible(r)]
        candidates.sort(
            key=lambda r: (r[score], r["iomin"], r["purity"], r["iou"]), reverse=True
        )
        pi, gi = set(), set()
        for r in candidates:
            if r["i"] not in pi and r["j"] not in gi:
                pi.add(r["i"])
                gi.add(r["j"])
        return len(pi)

    out = {
        "reference_count": m,
        "raw_prediction_count": len(predictions),
        "evaluable_prediction_count": n,
        "status": "conditional_frequency_support",
    }
    criteria = [
        ("iou50", "iou", lambda r: r["iou"] >= 0.5),
        ("iomin50", "iomin", lambda r: r["iomin"] >= 0.5),
        ("temporal_iomin50", "temporal_iomin", lambda r: r["temporal_iomin"] >= 0.5),
        (
            "t_gated",
            "temporal_iomin",
            lambda r: r["temporal_iomin"] >= 0.5
            and r["frequency_iomin"] >= 0.2
            and r["purity"] >= 0.25,
        ),
    ]
    for name, score, eligible in criteria:
        tp = match(score, eligible)
        fp = n - tp
        fn = m - tp
        out.update(
            {
                f"{name}_tp": tp,
                f"{name}_fp": fp,
                f"{name}_fn": fn,
                f"{name}_precision": tp / n if n else 0.0,
                f"{name}_recall": tp / m,
                f"{name}_f1": 2 * tp / (n + m) if n + m else 0.0,
            }
        )
    out["mean_best_temporal_iomin"] = float(
        np.mean(
            [
                max((r["temporal_iomin"] for r in pairs if r["j"] == j), default=0.0)
                for j in range(m)
            ]
        )
    )
    return out
