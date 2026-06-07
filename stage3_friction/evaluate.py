import argparse


def main(args):
    print("Evaluation stub: computes ROC-AUC and Macro-F1 from predictions.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preds', required=False)
    args = p.parse_args()
    main(args)
