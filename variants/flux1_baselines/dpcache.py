"""Run official DPCache on the existing FLUX.1-dev DrawBench experiment."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    parser.add_argument("source", type=Path, help="Existing 50-step FLUX.1-dev run")
    parser.add_argument("output", type=Path)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cost-matrix", type=Path)
    args = parser.parse_args()

    import torch
    from PIL import Image, ImageChops
    from ecad.image_generators.flux_image_generator import FluxImageGenerator
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'baselines/DPCache'))
    from dpcache_flux import apply_dpcache_to_flux
    from dpcache.cali_utils import merge_calibration_results
    from reactive_samplers.qwen_metrics import LegacyMetrics

    source, output = args.source.resolve(), args.output.resolve()
    cfg = json.loads((source / "config.json").read_text())
    assert cfg["steps"] == 50 and cfg["height"] == cfg["width"] == 256
    assert cfg["guidance_scale"] == 5
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    save(output / "config.json", {**cfg, "method": "DPCache", "k": args.k,
                                  "source_run": str(source)})

    FluxImageGenerator.DEFAULT_NUM_INFERENCE_STEPS = cfg["steps"]
    generator = FluxImageGenerator(weights_name=cfg["model"])
    generator.pipeline_weights = cfg["model"]
    generator.create_encoder_pipeline()
    generator.create_diffusion_pipeline()
    generator.diffusion_pipeline.set_progress_bar_config(disable=True)

    calibration = args.phase == "calibrate"
    cost_file = args.cost_matrix.resolve() if args.cost_matrix else output / "final_cost_matrix_flux1.pkl"
    if not calibration and not cost_file.is_file():
        raise FileNotFoundError(f"Calibrate first or pass --cost-matrix: {cost_file}")
    apply_dpcache_to_flux(generator.diffusion_pipeline, cali=calibration,
                          num_steps=50, order=2, k=args.k,
                          first_full_steps=3, cali_prefix="flux1",
                          cost_matrix_path=str(cost_file))
    rows = cfg["rows"][:10] if calibration else cfg["rows"][:args.limit or None]
    records_file = output / "results.partial.json"
    records = json.loads(records_file.read_text()) if not calibration and records_file.exists() else []
    completed = {r["id"] for r in records}
    metrics = None if calibration else LegacyMetrics(cfg["metrics_python"],
                                                      {"names": ["psnr", "ssim", "lpips"]})
    try:
        for row in rows:
            if row["id"] in completed:
                continue
            generator.start_seed = row["seed"]
            torch.cuda.synchronize()
            start = time.perf_counter()
            embed = generator.encode_prompts([row["prompt"]], batch_size=1)[0]
            embed = {key: value.unsqueeze(0) if torch.is_tensor(value) else value
                     for key, value in embed.items()}
            image = generator.generate_images(embed, height=256, width=256,
                                               guidance_scale=5)[0][0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            reference = source / "original" / (row["id"] + ".png")
            if calibration:
                if row == rows[0]:
                    difference = ImageChops.difference(image.convert("RGB"),
                                                       Image.open(reference).convert("RGB"))
                    assert difference.getbbox() is None, "Full-compute DPCache differs from original"
                    print("FULL-COMPUTE CONTROL PASSED", row["id"], flush=True)
            else:
                image_path = output / "images" / (row["id"] + ".png")
                image_path.parent.mkdir(exist_ok=True)
                image.save(image_path)
                record = {**row, "method": "dpcache", "image": str(image_path),
                          "seconds": seconds, "nfe": args.k,
                          "scores": metrics.score(image_path, row["prompt"], reference)}
                records.append(record)
                save(records_file, records)
                save(output / "status.json", {"status": "generating", "k": args.k,
                                               "completed": len(records), "total": len(cfg["rows"])})
                print(row["id"], "seconds", round(seconds, 3), "completed", len(records), flush=True)
    finally:
        if metrics is not None:
            metrics.close()

    if calibration:
        assert merge_calibration_results("flux1") == cost_file.name
        save(output / "status.json", {"status": "calibrated", "count": len(rows),
                                       "full_compute_control": "pixel_identical"})
        return

    summary = {"count": len(records), "mean_seconds": statistics.mean(r["seconds"] for r in records),
               "mean_nfe": args.k}
    for metric in ("psnr", "ssim", "lpips"):
        summary["mean_" + metric] = statistics.mean(r["scores"][metric] for r in records)
    original = json.loads((source / "results.json").read_text())["summary"]["original"]
    summary["speedup"] = original["mean_seconds"] / summary["mean_seconds"]
    save(output / "generation_results.json", {"summary": {"dpcache": summary}, "images": records})
    save(output / "status.json", {"status": "completed", "summary": summary})
    print("DPCACHE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
