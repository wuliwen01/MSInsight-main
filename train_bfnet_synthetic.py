import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model_bfnet import StrikeDipRakeNet
from synthetic_bfnet_utils import (
    BFNET_SHAPE,
    ensure_dir,
    load_state_dict_flexible,
    minmax_normalize_tensor,
    read_csv_records,
    resolve_record_path,
    sdr_to_moment_tensor_torch,
    sincos_to_moment_tensor_torch,
    write_json,
)


class BrightnessFieldDataset(Dataset):
    def __init__(self, samples_csv):
        self.samples_csv = samples_csv
        self.records = read_csv_records(samples_csv)
        if not self.records:
            raise ValueError(f"No records found in {samples_csv}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = np.load(resolve_record_path(self.samples_csv, rec["input_path"])).astype(np.float32)
        sdr = np.array([float(rec["strike"]), float(rec["dip"]), float(rec["rake"])], dtype=np.float32)
        return torch.from_numpy(arr), torch.from_numpy(sdr)


def split_dataset(dataset, val_ratio, seed):
    n_val = max(1, int(round(len(dataset) * val_ratio)))
    n_train = len(dataset) - n_val
    if n_train <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def init_output_bias(model):
    with torch.no_grad():
        bias = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], dtype=model.fcfinal.bias.dtype)
        model.fcfinal.bias.copy_(bias)


def run_epoch(model, loader, optimizer, device, dip_penalty_weight):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_items = 0
    for x, sdr in loader:
        x = x.to(device, non_blocking=True)
        sdr = sdr.to(device, non_blocking=True)
        x = minmax_normalize_tensor(x)

        with torch.set_grad_enabled(optimizer is not None):
            output = model(x)
            pred_mt = sincos_to_moment_tensor_torch(output)
            true_mt = sdr_to_moment_tensor_torch(sdr)
            loss = F.mse_loss(pred_mt, true_mt)
            if dip_penalty_weight > 0:
                dip_penalty = F.relu(-output[:, 2]).mean() + F.relu(-output[:, 3]).mean()
                loss = loss + dip_penalty_weight * dip_penalty

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def save_state_dict(model, output_path):
    ensure_dir(Path(output_path).parent)
    torch.save(model.state_dict(), output_path)


def append_log(log_path, row):
    ensure_dir(Path(log_path).parent)
    exists = Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "epoch", "train_loss", "val_loss", "lr"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_stage(stage_name, model, train_loader, val_loader, optimizer, scheduler, epochs, args, device, best):
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, args.dip_penalty_weight)
        val_loss = run_epoch(model, val_loader, None, device, args.dip_penalty_weight)
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        append_log(
            args.log_csv,
            {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": f"{train_loss:.8f}",
                "val_loss": f"{val_loss:.8f}",
                "lr": f"{lr:.8g}",
            },
        )
        print(f"{stage_name} epoch {epoch:03d}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} lr={lr:.3g}")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "stage": stage_name, "epoch": epoch}
            save_state_dict(model, args.output)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "shape": BFNET_SHAPE,
                    "best": best,
                    "args": vars(args),
                },
                args.checkpoint,
            )
            print(f"  saved best model to {args.output}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Train BFNet on synthetic brightness-field .npy samples.")
    parser.add_argument("--samples-csv", default="synthetic/bfnet_samples/samples.csv")
    parser.add_argument("--output", default="model/bfnet_synthetic.pt")
    parser.add_argument("--checkpoint", default="model/bfnet_synthetic.ckpt")
    parser.add_argument("--log-csv", default="result/bfnet_synthetic_train_log.csv")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stage1-epochs", type=int, default=200)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--stage1-batch-size", type=int, default=16)
    parser.add_argument("--stage2-batch-size", type=int, default=8)
    parser.add_argument("--stage1-lr", type=float, default=5e-4)
    parser.add_argument("--stage1-momentum", type=float, default=0.0)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-min-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dip-penalty-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataset = BrightnessFieldDataset(args.samples_csv)
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)

    model = StrikeDipRakeNet(BFNET_SHAPE).to(device)
    init_output_bias(model)
    if args.resume:
        model.load_state_dict(load_state_dict_flexible(args.resume, map_location=device))
        print(f"Resumed model from {args.resume}")

    best = {"val_loss": float("inf"), "stage": None, "epoch": None}

    if args.stage1_epochs > 0:
        train_loader = DataLoader(
            train_set,
            batch_size=args.stage1_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.stage1_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=args.stage1_lr, momentum=args.stage1_momentum)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 25, 50, 100, 150], gamma=0.5
        )
        best = train_stage("stage1_sgd", model, train_loader, val_loader, optimizer, scheduler, args.stage1_epochs, args, device, best)

    if args.stage2_epochs > 0:
        train_loader = DataLoader(
            train_set,
            batch_size=args.stage2_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.stage2_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.stage2_epochs, eta_min=args.stage2_min_lr
        )
        best = train_stage("stage2_adamw", model, train_loader, val_loader, optimizer, scheduler, args.stage2_epochs, args, device, best)

    write_json(Path(args.output).with_suffix(".json"), {"best": best, "args": vars(args)})
    print(f"Best validation loss: {best['val_loss']:.6f} ({best['stage']} epoch {best['epoch']})")


if __name__ == "__main__":
    main()
