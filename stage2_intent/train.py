import argparse
import torch
from torch.utils.data import DataLoader
from model import GRUIntentModel
from dataset import ClickstreamDataset


def main(args):
    print("Stage 2 training stub")
    ds = ClickstreamDataset([], seq_len=args.seq_len)
    dl = DataLoader(ds, batch_size=8)
    model = GRUIntentModel(input_dim=32, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        for x, y in dl:
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Finished epoch {epoch+1}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=False)
    p.add_argument('--seq_len', type=int, default=20)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--epochs', type=int, default=1)
    args = p.parse_args()
    main(args)
