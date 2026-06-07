import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args):
    print(f"DOM embeddings stub. Reading from: {args.input}")
    os.makedirs(args.output, exist_ok=True)
    # Placeholder: create empty embeddings file
    df = pd.DataFrame({"task_id": [], "dom_embedding": []})
    out_path = os.path.join(args.output, "dom_embeddings.parquet")
    df.to_parquet(out_path)
    print(f"Wrote placeholder: {out_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    main(args)
