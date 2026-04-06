import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--processed-name", type=str, default="cora")
    parser.add_argument("--split-idx", type=int, default=None)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    metrics = evaluate_checkpoint(
        str(ckpt),
        processed_name=args.processed_name,
        split_idx_override=args.split_idx,
    )
    print(metrics)


if __name__ == "__main__":
    main()
