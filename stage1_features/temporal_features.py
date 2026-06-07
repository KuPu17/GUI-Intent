import argparse
import os
import pandas as pd


def main(args):
    print(f"Temporal features stub. Reading from: {args.input}")
    os.makedirs(args.output, exist_ok=True)
    df = pd.DataFrame({"task_id": [], "delta_t": []})
    out_path = os.path.join(args.output, "temporal_features.parquet")
    df.to_parquet(out_path)
    print(f"Wrote placeholder: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    main(args)
