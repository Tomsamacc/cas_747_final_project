import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np


def _normalize_dataset_name(ds: str) -> str:
    s = ds.lower().replace("_", "-")
    if s == "arxiv_year":
        s = "arxiv-year"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="use the last N summary lines (e.g. 10 splits, or 5 for arxiv-year)",
    )
    args = parser.parse_args()

    name = _normalize_dataset_name(args.dataset)
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    grandparent = os.path.dirname(parent)
    logs_path = os.path.join(grandparent, "results", "logs", name, "train.log")

    if not os.path.isfile(logs_path):
        print(f"no log file: {logs_path}", file=sys.stderr)
        sys.exit(1)

    text = Path(logs_path).read_text(encoding="utf-8")

    pat_acc = re.compile(
        rf"dataset={re.escape(name)} \| best_val_acc ckpt -> val_acc=([\d.]+) test_acc=([\d.]+)"
    )
    pat_old = re.compile(
        rf"dataset={re.escape(name)} val_acc=([\d.]+) test_acc=([\d.]+)"
    )

    rows = [(float(a), float(b)) for a, b in pat_acc.findall(text)]
    if not rows:
        rows = [(float(a), float(b)) for a, b in pat_old.findall(text)]

    if not rows:
        print(
            f"no summary lines matched for dataset={name!r} in {logs_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = rows[-args.last_n :]
    va, ta = zip(*rows)

    pat_t3 = re.compile(
        rf"dataset={re.escape(name)} full_wall_time_sec=([\d.]+) precompute_wall_time_sec=([\d.]+) training_cost_time_sec=([\d.]+) \| best_val_loss"
    )
    pat_t1 = re.compile(
        rf"dataset={re.escape(name)} time=([\d.]+)s \| best_val_loss"
    )

    trows = [(float(a), float(b), float(c)) for a, b, c in pat_t3.findall(text)]
    if not trows:
        trows = [(float(a), np.nan, np.nan) for a in pat_t1.findall(text)]

    if trows:
        trows = trows[-args.last_n :]

    print(f"parsed {len(rows)} runs from {logs_path} (best_val_acc ckpt metrics)")
    print(f"val  mean={np.mean(va)*100:.2f}% std={np.std(va)*100:.2f}%")
    print(f"test mean={np.mean(ta)*100:.2f}% std={np.std(ta)*100:.2f}%")

    if not trows:
        print("time: no matching lines (need best_val_loss line with full_wall_time_sec or time=)")
        return

    if len(trows) != len(rows):
        print(
            f"time: parsed {len(trows)} lines vs {len(rows)} acc lines (last-n each); stats use time rows only",
            file=sys.stderr,
        )

    fw, pr, tr = zip(*trows)
    print(
        f"full_wall_time_sec     mean={np.nanmean(fw):.3f}s std={np.nanstd(fw, ddof=0):.3f}s (n={len(trows)})"
    )
    if not all(np.isnan(pr)):
        print(
            f"precompute_wall_time_sec mean={np.nanmean(pr):.3f}s std={np.nanstd(pr, ddof=0):.3f}s"
        )
    if not all(np.isnan(tr)):
        print(
            f"training_cost_time_sec   mean={np.nanmean(tr):.3f}s std={np.nanstd(tr, ddof=0):.3f}s"
        )


if __name__ == "__main__":
    main()
