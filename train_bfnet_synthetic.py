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
    decode_sincos_np,
    encode_sdr_sincos_np,
    ensure_dir,
    focal_planes,
    load_state_dict_flexible,
    mechanism_error_componentwise,
    minmax_normalize_tensor,
    read_csv_records,
    resolve_record_path,
    sdr_to_moment_tensor_torch,
    sincos_to_moment_tensor_torch,
    write_json,
)


LOG_FIELDS = [
    "stage",
    "epoch",
    "train_loss",
    "val_loss",
    "train_mt_loss",
    "val_mt_loss",
    "train_aux_loss",
    "val_aux_loss",
    "lr",
    "val_strike_mean_deg",
    "val_strike_median_deg",
    "val_dip_mean_deg",
    "val_dip_median_deg",
    "val_rake_mean_deg",
    "val_rake_median_deg",
    "val_mechanism_score_deg",
]


class BrightnessFieldDataset(Dataset):
    def __init__(self, samples_csv, include_equiv_sincos=False):
        self.samples_csv = samples_csv
        self.records = read_csv_records(samples_csv)
        if not self.records:
            raise ValueError(f"No records found in {samples_csv}")
        self.equiv_sincos_targets = None
        if include_equiv_sincos:
            self.equiv_sincos_targets = self._build_equiv_sincos_targets()

    def _build_equiv_sincos_targets(self):
        targets = []
        for rec in self.records:
            sdr = (float(rec["strike"]), float(rec["dip"]), float(rec["rake"]))
            planes = focal_planes(*sdr)
            if len(planes) < 2:
                planes = planes + planes[:1]
            targets.append([encode_sdr_sincos_np(*plane) for plane in planes[:2]])
        return np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = np.load(resolve_record_path(self.samples_csv, rec["input_path"])).astype(np.float32)
        sdr = np.array([float(rec["strike"]), float(rec["dip"]), float(rec["rake"])], dtype=np.float32)
        if self.equiv_sincos_targets is not None:
            return torch.from_numpy(arr), torch.from_numpy(sdr), torch.from_numpy(self.equiv_sincos_targets[idx])
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


def unpack_batch(batch):
    if len(batch) == 3:
        return batch
    x, sdr = batch
    return x, sdr, None


def equiv_sincos_loss(output, equiv_targets):
    per_plane_mse = (output[:, None, :] - equiv_targets).pow(2).mean(dim=2)
    return per_plane_mse.min(dim=1).values.mean()


def run_epoch(model, loader, optimizer, device, aux_weight=0.0):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_mt_loss = 0.0
    total_aux_loss = 0.0
    total_items = 0
    for batch in loader:
        x, sdr, equiv_targets = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        sdr = sdr.to(device, non_blocking=True)
        if equiv_targets is not None:
            equiv_targets = equiv_targets.to(device, non_blocking=True)
        x = minmax_normalize_tensor(x)

        with torch.set_grad_enabled(optimizer is not None):
            output = model(x)
            pred_mt = sincos_to_moment_tensor_torch(output)
            true_mt = sdr_to_moment_tensor_torch(sdr)
            mt_loss = F.mse_loss(pred_mt, true_mt)
            aux_loss = output.new_tensor(0.0)
            if aux_weight > 0.0 and equiv_targets is not None:
                aux_loss = equiv_sincos_loss(output, equiv_targets)
            loss = mt_loss + aux_weight * aux_loss

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mt_loss += float(mt_loss.detach().cpu()) * batch_size
        total_aux_loss += float(aux_loss.detach().cpu()) * batch_size
        total_items += batch_size
    total_items = max(total_items, 1)
    return {
        "loss": total_loss / total_items,
        "mt_loss": total_mt_loss / total_items,
        "aux_loss": total_aux_loss / total_items,
    }


def evaluate_mechanism_errors(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            x, sdr, _ = unpack_batch(batch)
            x = minmax_normalize_tensor(x.to(device, non_blocking=True))
            output = model(x).cpu().numpy()
            pred_sdr = decode_sincos_np(output)
            true_sdr = sdr.cpu().numpy()
            for pred, true in zip(pred_sdr, true_sdr):
                errors.append(mechanism_error_componentwise(pred, true))

    if not errors:
        return None

    errors = np.asarray(errors, dtype=np.float32)
    means = np.mean(errors, axis=0)
    medians = np.median(errors, axis=0)
    return {
        "val_strike_mean_deg": float(means[0]),
        "val_strike_median_deg": float(medians[0]),
        "val_dip_mean_deg": float(means[1]),
        "val_dip_median_deg": float(medians[1]),
        "val_rake_mean_deg": float(means[2]),
        "val_rake_median_deg": float(medians[2]),
    }


def mechanism_score(metrics):
    return (
        metrics["val_strike_median_deg"]
        + metrics["val_dip_median_deg"]
        + metrics["val_rake_median_deg"]
    )


def save_state_dict(model, output_path):
    ensure_dir(Path(output_path).parent)
    torch.save(model.state_dict(), output_path)


def save_checkpoint(model, checkpoint_path, best, args):
    ensure_dir(Path(checkpoint_path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "shape": BFNET_SHAPE,
            "best": best,
            "args": vars(args),
        },
        checkpoint_path,
    )


def add_stem_suffix(path, suffix):
    path = Path(path)
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def append_log(log_path, row):
    path = Path(log_path)
    ensure_dir(path.parent)
    normalized_row = {field: row.get(field, "") for field in LOG_FIELDS}

    write_header = not path.exists() or path.stat().st_size == 0
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            existing_rows = list(reader)
        if existing_fields != LOG_FIELDS:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({field: existing_row.get(field, "") for field in LOG_FIELDS})
        write_header = False

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def train_stage(stage_name, model, train_loader, val_loader, optimizer, scheduler, epochs, args, device, best_loss, best_mechanism):
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.equiv_sincos_loss_weight)
        val_metrics = run_epoch(model, val_loader, None, device, args.equiv_sincos_loss_weight)
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        mechanism_metrics = evaluate_mechanism_errors(model, val_loader, device) if args.log_mechanism_errors else None
        current_mechanism_score = mechanism_score(mechanism_metrics) if mechanism_metrics is not None else None
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        log_row = {
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": f"{train_loss:.8f}",
            "val_loss": f"{val_loss:.8f}",
            "train_mt_loss": f"{train_metrics['mt_loss']:.8f}",
            "val_mt_loss": f"{val_metrics['mt_loss']:.8f}",
            "train_aux_loss": f"{train_metrics['aux_loss']:.8f}",
            "val_aux_loss": f"{val_metrics['aux_loss']:.8f}",
            "lr": f"{lr:.8g}",
        }
        if mechanism_metrics is not None:
            log_row.update({key: f"{value:.6f}" for key, value in mechanism_metrics.items()})
            log_row["val_mechanism_score_deg"] = f"{current_mechanism_score:.6f}"
        append_log(args.log_csv, log_row)

        message = f"{stage_name} epoch {epoch:03d}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} lr={lr:.3g}"
        if args.equiv_sincos_loss_weight > 0.0:
            message += f" mt_val={val_metrics['mt_loss']:.6f} aux_val={val_metrics['aux_loss']:.6f}"
        if mechanism_metrics is not None:
            message += (
                " mech_med="
                f"({mechanism_metrics['val_strike_median_deg']:.2f},"
                f"{mechanism_metrics['val_dip_median_deg']:.2f},"
                f"{mechanism_metrics['val_rake_median_deg']:.2f}) deg"
            )
        print(message)

        if val_loss < best_loss["val_loss"]:
            best_loss = {"val_loss": val_loss, "stage": stage_name, "epoch": epoch}
            save_state_dict(model, args.output)
            save_checkpoint(model, args.checkpoint, best_loss, args)
            print(f"  saved best model to {args.output}")

        if current_mechanism_score is not None and current_mechanism_score < best_mechanism["val_mechanism_score_deg"]:
            best_mechanism = {
                "val_mechanism_score_deg": current_mechanism_score,
                "stage": stage_name,
                "epoch": epoch,
                "val_loss": val_loss,
                **mechanism_metrics,
            }
            save_state_dict(model, args.mechanism_output)
            save_checkpoint(model, args.mechanism_checkpoint, best_mechanism, args)
            print(f"  saved best mechanism model to {args.mechanism_output}")
    return best_loss, best_mechanism


def main():
    parser = argparse.ArgumentParser(description="Train BFNet on synthetic brightness-field .npy samples.")
    parser.add_argument("--samples-csv", default="synthetic/bfnet_samples/samples.csv")
    parser.add_argument("--output", default="model/bfnet_synthetic.pt")
    parser.add_argument("--checkpoint", default="model/bfnet_synthetic.ckpt")
    parser.add_argument("--mechanism-output", default=None)
    parser.add_argument("--mechanism-checkpoint", default=None)
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
    parser.add_argument(
        "--equiv-sincos-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary loss weight. When >0, the total training loss is "
            "MT MSE plus this weight times the smaller sin/cos MSE to the true "
            "fault plane or its auxiliary plane."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--log-mechanism-errors",
        action="store_true",
        help="Log validation strike/dip/rake errors for every epoch using the thesis componentwise equivalent-plane metric.",
    )
    args = parser.parse_args()
    if args.log_mechanism_errors:
        if args.mechanism_output is None:
            args.mechanism_output = add_stem_suffix(args.output, "_best_mech")
        if args.mechanism_checkpoint is None:
            args.mechanism_checkpoint = add_stem_suffix(args.checkpoint, "_best_mech")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dataset = BrightnessFieldDataset(args.samples_csv, include_equiv_sincos=args.equiv_sincos_loss_weight > 0.0)
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)

    model = StrikeDipRakeNet(BFNET_SHAPE).to(device)
    init_output_bias(model)
    if args.resume:
        model.load_state_dict(load_state_dict_flexible(args.resume, map_location=device))
        print(f"Resumed model from {args.resume}")

    best = {"val_loss": float("inf"), "stage": None, "epoch": None}
    best_mechanism = {"val_mechanism_score_deg": float("inf"), "stage": None, "epoch": None}

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
        best, best_mechanism = train_stage(
            "stage1_sgd",
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            args.stage1_epochs,
            args,
            device,
            best,
            best_mechanism,
        )

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
        best, best_mechanism = train_stage(
            "stage2_adamw",
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            args.stage2_epochs,
            args,
            device,
            best,
            best_mechanism,
        )

    payload = {"best": best, "args": vars(args)}
    if args.log_mechanism_errors:
        payload["best_mechanism"] = best_mechanism
    write_json(Path(args.output).with_suffix(".json"), payload)
    if args.log_mechanism_errors:
        write_json(Path(args.mechanism_output).with_suffix(".json"), {"best": best_mechanism, "best_loss": best, "args": vars(args)})
    print(f"Best validation loss: {best['val_loss']:.6f} ({best['stage']} epoch {best['epoch']})")
    if args.log_mechanism_errors:
        print(
            "Best mechanism score: "
            f"{best_mechanism['val_mechanism_score_deg']:.6f} "
            f"({best_mechanism['stage']} epoch {best_mechanism['epoch']})"
        )


if __name__ == "__main__":
    main()
