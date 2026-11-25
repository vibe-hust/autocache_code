"""Paired frame metrics against the same-seed official HunyuanVideo output."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def score(reference_path, candidate_path, model, device):
    reference = np.asarray(imageio.mimread(reference_path), dtype=np.uint8)
    candidate = np.asarray(imageio.mimread(candidate_path), dtype=np.uint8)
    if reference.shape != candidate.shape:
        raise ValueError(f"Video shapes differ: {reference.shape} and {candidate.shape}")
    psnr = []
    ssim = []
    for a, b in zip(reference, candidate):
        psnr.append(float(peak_signal_noise_ratio(a, b, data_range=255)))
        ssim.append(float(structural_similarity(a, b, channel_axis=-1, data_range=255)))
    values = []
    with torch.no_grad():
        for start in range(0, len(reference), 4):
            a = torch.from_numpy(reference[start:start + 4].copy()).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
            b = torch.from_numpy(candidate[start:start + 4].copy()).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
            values.extend(model(a, b).flatten().cpu().tolist())
    mean_psnr = float(np.mean(psnr))
    return {"frames": len(reference), "psnr": mean_psnr if np.isfinite(mean_psnr) else None,
            "ssim": float(np.mean(ssim)), "lpips": float(np.mean(values))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_json", help="List of {id, reference, candidate} rows")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path = Path(args.pairs_json)
    pairs = json.loads(path.read_text(encoding="utf-8"))
    model = lpips.LPIPS(net="alex", verbose=False).eval().to(args.device)
    rows = [{"id": row["id"], **score(row["reference"], row["candidate"], model, args.device)}
            for row in pairs]
    summary = {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
               if any(row[key] is not None for row in rows) else None
               for key in ("psnr", "ssim", "lpips")}
    result = {"per_video": rows, "average": summary}
    path.with_name("paired_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
