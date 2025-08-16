"""
Overfit a tiny synthetic dataset to prove your MLP + training loop.
"""
import argparse, torch
from torch.utils.data import DataLoader, TensorDataset
from .mlp import MLP
from .train_loop import train_one_epoch, eval_loss

def make_synth(n=512, in_dim=16, num_classes=3, seed=123):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, in_dim, generator=g)
    W = torch.randn(in_dim, num_classes, generator=g) * 0.5
    b = torch.randn(num_classes, generator=g) * 0.1
    logits = X @ W + b
    y = logits.argmax(dim=1)
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--in_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_hidden", type=int, default=2)
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    X, y = make_synth(n=512, in_dim=args.in_dim, num_classes=args.num_classes)
    dl = DataLoader(TensorDataset(X, y), batch_size=args.batch_size, shuffle=True)

    model = MLP(args.in_dim, args.hidden_dim, args.num_classes, num_hidden=args.num_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, dl, opt)
        ev_loss = eval_loss(model, dl)
        print(f"epoch {epoch:02d} | train {tr_loss:.4f} | eval {ev_loss:.4f}")

if __name__ == "__main__":
    main()
