import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model_bfnet import StrikeDipRakeNet
from synthetic_bfnet_utils import (
    BFNET_SHAPE,
    decode_sincos_np,
    ensure_dir,
    load_state_dict_flexible,
    mechanism_error,
    minmax_normalize_tensor,
    read_csv_records,
    resolve_record_path,
    write_json,
)


class EvalDataset(Dataset):
    def __init__(self, samples_csv):
        self.samples_csv = samples_csv
        self.records = read_csv_records(samples_csv)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = np.load(resolve_record_path(self.samples_csv, rec["input_path"])).astype(np.float32)
        true_sdr = np.array([float(rec["strike"]), float(rec["dip"]), float(rec["rake"])], dtype=np.float32)
        return torch.from_numpy(arr), torch.from_numpy(true_sdr), idx


def summarize(errors):
    errors = np.asarray(errors, dtype=np.float32)
    return {
        "strike_mean": float(np.mean(errors[:, 0])),
        "strike_median": float(np.median(errors[:, 0])),
        "dip_mean": float(np.mean(errors[:, 1])),
        "dip_median": float(np.median(errors[:, 1])),
        "rake_mean": float(np.mean(errors[:, 2])),
        "rake_median": float(np.median(errors[:, 2])),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a BFNet checkpoint on synthetic brightness-field .npy samples.")
    parser.add_argument("--samples-csv", default="synthetic/bfnet_samples/samples.csv")
    parser.add_argument("--model", default="model/bfnet_synthetic.pt")
    parser.add_argument("--predictions-csv", default="result/bfnet_synthetic_predictions.csv")
    parser.add_argument("--metrics-json", default="result/bfnet_synthetic_metrics.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = EvalDataset(args.samples_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = StrikeDipRakeNet(BFNET_SHAPE).to(device)
    model.load_state_dict(load_state_dict_flexible(args.model, map_location=device))
    model.eval()

    predictions = []
    bfnet_errors = []
    jssa_errors = []
    with torch.no_grad():
        for x, true_sdr, indices in loader:
            x = minmax_normalize_tensor(x.to(device, non_blocking=True))
            output = model(x).cpu().numpy()
            pred_sdr = decode_sincos_np(output)
            true_np = true_sdr.numpy()
            for local_i, rec_idx in enumerate(indices.numpy().tolist()):
                rec = dataset.records[rec_idx]
                pred = tuple(float(v) for v in pred_sdr[local_i])
                true = tuple(float(v) for v in true_np[local_i])
                err = mechanism_error(pred, true)
                bfnet_errors.append(err)

                row = {
                    "event_id": rec["event_id"],
                    "true_strike": true[0],
                    "true_dip": true[1],
                    "true_rake": true[2],
                    "bfnet_strike": pred[0],
                    "bfnet_dip": pred[1],
                    "bfnet_rake": pred[2],
                    "bfnet_err_strike": err[0],
                    "bfnet_err_dip": err[1],
                    "bfnet_err_rake": err[2],
                }
                if "jssa_strike" in rec and rec["jssa_strike"] != "":
                    jssa = (float(rec["jssa_strike"]), float(rec["jssa_dip"]), float(rec["jssa_rake"]))
                    jerr = mechanism_error(jssa, true)
                    jssa_errors.append(jerr)
                    row.update(
                        {
                            "jssa_strike": jssa[0],
                            "jssa_dip": jssa[1],
                            "jssa_rake": jssa[2],
                            "jssa_err_strike": jerr[0],
                            "jssa_err_dip": jerr[1],
                            "jssa_err_rake": jerr[2],
                        }
                    )
                predictions.append(row)

    ensure_dir(Path(args.predictions_csv).parent)
    fieldnames = list(predictions[0].keys()) if predictions else []
    with open(args.predictions_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {"bfnet": summarize(bfnet_errors)}
    if jssa_errors:
        metrics["jssa"] = summarize(jssa_errors)
    write_json(args.metrics_json, metrics)
    print(f"Predictions saved to {args.predictions_csv}")
    print(f"Metrics saved to {args.metrics_json}")
    print(metrics)


if __name__ == "__main__":
    main()

