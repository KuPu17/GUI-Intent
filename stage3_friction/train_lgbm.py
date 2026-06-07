import argparse
import os
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split


def main(args):
    print("LightGBM training stub")
    # In a real run, load features from args.data
    print("No data provided; this is a stub that demonstrates CLI and params.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=False)
    p.add_argument('--smote', default='true')
    p.add_argument('--eval_metric', default='macro_f1')
    args = p.parse_args()
    main(args)
