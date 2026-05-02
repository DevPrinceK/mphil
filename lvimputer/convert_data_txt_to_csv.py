import argparse
from pathlib import Path

import pandas as pd


COL_NAMES = [
    "date",
    "time",
    "epoch",
    "moteid",
    "temperature",
    "humidity",
    "light",
    "voltage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Intel Lab-style whitespace-delimited data.txt to data.csv"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data.txt",
        help="Path to input data.txt (default: data.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data.csv",
        help="Path to output CSV (default: data.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    df = pd.read_csv(
        input_path,
        sep=r"\s+",
        names=COL_NAMES,
        header=None,
        engine="python",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
