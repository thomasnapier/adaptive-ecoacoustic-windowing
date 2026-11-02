"""Commands for checking data, running detection, and creating reference plots."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="WABAD fine-band drift event proposals. See configs/paper_example.json for the ARD paper example."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("run", help="Detect candidate sound events")
    detect.add_argument("--config", required=True)
    detect.add_argument("--limit", type=int)
    detect.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and annotations without detection or writing outputs",
    )
    detect.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when a matching completion record exists",
    )
    plot = sub.add_parser("plot-gt", help="Draw labelled reference spectrograms")
    plot.add_argument("--config", required=True)
    plot.add_argument("--limit", type=int)
    args = parser.parse_args()
    try:
        if args.command == "run":
            from .pipeline import run

            run(args.config, args.limit, args.dry_run, args.force)
        else:
            from .config import load_config
            from .pipeline import records_for, validate_record
            from . import visualize

            cfg = load_config(args.config)
            records = records_for(cfg)
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("limit must be at least 1")
                records = records[: args.limit]
            visualize.FMAX_HZ = cfg["parameters"]["FMAX"]
            count = 0
            for r in records:
                refs, _ = validate_record(r, cfg["parameters"])
                if refs is None:
                    continue
                folder = Path(cfg["output_dir"]) / r["site"] / r["audio"].stem
                folder.mkdir(parents=True, exist_ok=True)
                rois = [(b["t0"], b["t1"], b["f0"], b["f1"], b["label"]) for b in refs]
                visualize.plot_spectrogram_with_rois(
                    r["audio"], rois, folder / "ground_truth.png", r["site"]
                )
                count += 1
            print(f"Saved {count} reference plot(s) under {cfg['output_dir']}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
