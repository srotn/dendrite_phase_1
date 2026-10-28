from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from experiments.dendrite.phase_1_synthetic_experiments.discrete_selection import LearnableDiscreteDD

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def log_uniform(g, low, high, n):
    return torch.exp(torch.empty(n).uniform_(math.log(low), math.log(high), generator=g))


def generate(n, seed):
    g = torch.Generator().manual_seed(seed)
    m = log_uniform(g, .5, 5., n); k = log_uniform(g, 10., 200., n)
    c = log_uniform(g, .03, 8., n); force = log_uniform(g, .1, 10., n)
    ratio = log_uniform(g, .3, 3., n); omega = ratio * torch.sqrt(k / m)
    amp = force / torch.sqrt((k - m * omega.square()).square() + (c * omega).square()).clamp_min(1e-12)
    phase = torch.atan2(c * omega, k - m * omega.square())
    physical = torch.stack((m, k, c, omega, force), 1)
    mins = torch.tensor([.5, 10., .03, .3 * math.sqrt(10./5.), .1])
    maxs = torch.tensor([5., 200., 8., 3. * math.sqrt(200./.5), 10.])
    x = .5 + (torch.log(physical) - torch.log(mins)) / (torch.log(maxs) - torch.log(mins))
    return {"physical": physical.float(), "x": x.float(), "y": torch.stack((torch.log(amp), phase), 1).float()}


class DDStack(nn.Module):
    def __init__(self, width):
        super().__init__(); self.layers = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(3))

    def forward(self, x):
        anchor = x; h = x
        for layer in self.layers: h = layer(h) * anchor
        return h


class BaseDD(nn.Module):
    name = "base_dd"
    def __init__(self): super().__init__(); self.dd = DDStack(5); self.output = nn.Linear(5, 2)
    def forward(self, x): return self.output(self.dd(x))


class ReciprocalSharedDD(nn.Module):
    name = "reciprocal_shared_dd"
    def __init__(self): super().__init__(); self.dd = DDStack(5); self.output = nn.Linear(5, 2)
    def forward(self, x):
        return self.output(self.dd(x) * self.dd(1. / x.clamp_min(1e-6)))


class ReciprocalConcatDD(nn.Module):
    name = "reciprocal_concat_dd"
    def __init__(self): super().__init__(); self.dd = DDStack(10); self.output = nn.Linear(10, 2)
    def forward(self, x): return self.output(self.dd(torch.cat((x, 1. / x.clamp_min(1e-6)), 1)))


class DuplicateConcatDD(nn.Module):
    """Parameter-matched control: doubles input width without new information."""
    name = "duplicate_concat_dd"
    def __init__(self): super().__init__(); self.dd = DDStack(10); self.output = nn.Linear(10, 2)
    def forward(self, x): return self.output(self.dd(torch.cat((x, x), 1)))


class DeepBaseDD(nn.Module):
    name = "deep_base_dd_16_layers"
    def __init__(self):
        super().__init__(); self.dd = nn.ModuleList(nn.Linear(5, 5, bias=False) for _ in range(16)); self.output = nn.Linear(5, 2)
    def forward(self, x):
        anchor = x; h = x
        for layer in self.dd: h = layer(h) * anchor
        return self.output(h)


class LearnableDiscreteInputDD(LearnableDiscreteDD):
    name = "learnable_discrete_dd"

    def __init__(self):
        super().__init__(input_width=5, layers=3, output_dim=2, name=self.name)


MODELS = (BaseDD, ReciprocalSharedDD, ReciprocalConcatDD, DuplicateConcatDD, DeepBaseDD, LearnableDiscreteInputDD)


def counts(): return {c.name: sum(p.numel() for p in c().parameters()) for c in MODELS}


def metrics(pred, target, mean, scale):
    actual = pred * scale + mean; truth = target * scale + mean
    ap = torch.exp(actual[:, 0]); at = torch.exp(truth[:, 0])
    ae = (ap - at).abs() / at.clamp_min(1e-12)
    pe = (actual[:, 1] - truth[:, 1]).abs()
    return {"mse_normalized": float((pred - target).square().mean()), "amplitude_relative_error": float(ae.mean()), "phase_mae_rad": float(pe.mean()), "phase_mae_deg": float(torch.rad2deg(pe).mean())}


def train_model(cls, train, valid, mean, scale, args):
    model = cls().to(args.device)
    if hasattr(model, "_seed"): model._seed = getattr(args, "seed", 0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(train["x"], train["y"]), batch_size=args.batch, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(TensorDataset(valid["x"], valid["y"]), batch_size=args.batch)
    folder = ARTIFACTS / cls.name; folder.mkdir(parents=True, exist_ok=True); best = float("inf")
    for epoch in range(1, args.epochs + 1):
        if hasattr(model, "set_epoch"): model.set_epoch(epoch, args.epochs)
        started = time.perf_counter(); model.train(); total = 0.; count = 0; norms = []
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device); opt.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean(); loss.backward()
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))); opt.step()
            total += float(loss.detach()) * len(x); count += len(x)
        model.eval(); pp=[]; yy=[]
        with torch.inference_mode():
            for x, y in val_loader: pp.append(model(x.to(args.device)).cpu()); yy.append(y)
        row = {"epoch": epoch, "seconds": time.perf_counter()-started, "train_loss": total/count, "gradient_norm": float(np.mean(norms)), **metrics(torch.cat(pp), torch.cat(yy), mean, scale)}
        if hasattr(model, "selection_values"): row["selection_values"] = model.selection_values().tolist()
        with (folder / "metrics.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        if row["mse_normalized"] < best:
            best = row["mse_normalized"]; checkpoint = {"model":model.state_dict(),"epoch":epoch,"metrics":row,"parameters":counts()[cls.name]}
            if hasattr(model, "selection_values"): checkpoint["selection_values"] = model.selection_values().tolist()
            torch.save(checkpoint, folder/"best.pt")
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs: print(json.dumps({"model":cls.name,**row}), flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--train-size",type=int,default=50000); p.add_argument("--valid-size",type=int,default=10000); p.add_argument("--epochs",type=int,default=500); p.add_argument("--batch",type=int,default=4096); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--seed",type=int,default=20261016); p.add_argument("--device",choices=("cpu","cuda"),default="cuda"); p.add_argument("--only",choices=[c.name for c in MODELS]); args=p.parse_args(); args.device=torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); ARTIFACTS.mkdir(parents=True,exist_ok=True)
    train=generate(args.train_size,args.seed); valid=generate(args.valid_size,args.seed+1); mean=train["y"].mean(0); scale=train["y"].std(0).clamp_min(1e-6); train["y"]=(train["y"]-mean)/scale; valid["y"]=(valid["y"]-mean)/scale
    torch.save({"train":train,"valid":valid,"y_mean":mean,"y_scale":scale},ARTIFACTS/"dataset.pt"); (ARTIFACTS/"parameter_report.json").write_text(json.dumps(counts(),indent=2),encoding="utf-8")
    (ARTIFACTS/"manifest.json").write_text(json.dumps({"seed":args.seed,"train_size":args.train_size,"valid_size":args.valid_size,"epochs":args.epochs,"models":[c.name for c in MODELS],"parameter_counts":counts(),"device":str(args.device)},indent=2),encoding="utf-8")
    for cls in MODELS:
        if args.only is None or args.only == cls.name: train_model(cls,train,valid,mean,scale,args)


if __name__ == "__main__": main()
