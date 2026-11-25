"""Run an upstream Wan entry on one of eight disjoint prompt shards."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from reactive_samplers.report import write_json
from variants.wan21_opal.evaluate import Evaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("method", choices=("official", "teacache", "dicache", "seacache"))
    parser.add_argument("shard", type=int, metavar="SHARD_INDEX")
    args = parser.parse_args()
    config = json.loads((args.root / "experiments/wan21-vbench124-820/baselines.json").read_text())
    config["weights"] = str((args.root/config["weights"]).resolve())
    config["reference_directory"] = str(args.experiment/"official")
    config["gpus_per_method"] = int(os.environ.get("SLURM_NTASKS",1))
    rows = [json.loads(line) for line in (args.root/"experiments/wan21-vbench124-820/test.jsonl").read_text().splitlines() if line.strip()][args.shard::config["gpus_per_method"]]
    out = args.experiment / args.method
    out.mkdir(parents=True,exist_ok=True)
    official = args.root/"baselines/Wan2.1"
    source, script = {
        "official": (official,"generate.py"),
        "teacache": (args.root/"baselines/TeaCache/TeaCache4Wan2.1","teacache_generate.py"),
        "dicache": (args.root/"baselines/DiCache/WAN2.1","run_wan_dicache.py"),
        "seacache": (args.root/"baselines/SeaCache/Wan2.1","seacache_generate.py"),
    }[args.method]
    sys.path[:0] = [str(source),str(official)]
    spec = importlib.util.spec_from_file_location("wan_upstream_entry", source / script)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import torch

    # Keep one model resident per GPU. The unchanged upstream generate() entry
    # initializes its cache state again before generating each new video.
    pipeline_type = entry.wan.WanT2V
    pipeline = None
    full_predictions = 0
    generation_seconds = 0.
    scorer = Evaluator({'frames':81}, 1, 64*1024*1024)
    scoring = ThreadPoolExecutor(max_workers=1)
    pending_scores = []

    def score_and_save(record, metadata):
        reference = str(Path(config['reference_directory'])/(record['id']+'.mp4'))
        record['scores'] = scorer.score(reference, record['video'])
        write_json(metadata, record)
        return record

    def count_backbone(module, inputs, output):
        nonlocal full_predictions
        full_predictions += 1

    def load_once(**kwargs):
        nonlocal pipeline
        if pipeline is None:
            pipeline = pipeline_type(**kwargs)
            # A full prediction reaches the last block; cached reuse does not.
            # Wan CFG evaluates conditional and unconditional branches separately.
            pipeline.model.blocks[-1].register_forward_hook(count_backbone)

            def timed_generate(*call_args, **call_kwargs):
                nonlocal generation_seconds
                torch.cuda.synchronize()
                began = time.perf_counter()
                # Upstream installs SeaCache's generate() on the class after load_once.
                video = type(pipeline).generate(pipeline, *call_args, **call_kwargs)
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter() - began
                return video

            pipeline.generate = timed_generate
        return pipeline

    entry.wan.WanT2V = load_once
    for row in rows:
        video = out / (row["id"] + ".mp4")
        metadata = video.with_suffix(".json")
        if metadata.exists():
            previous = json.loads(metadata.read_text())
            assert all(previous[k] == v for k, v in row.items())
            assert previous["method"] == args.method and previous["steps"] == config["steps"]
            assert video.stat().st_size > 0
            if 'scores' not in previous:
                pending_scores.append(scoring.submit(score_and_save, previous, metadata))
            continue
        pending = video.with_suffix(".pending.mp4")
        sys.argv = [str(source / script), "--task", "t2v-1.3B", "--size", "832*480",
                    "--ckpt_dir", config["weights"], "--prompt", row["prompt"],
                    "--base_seed", str(row["seed"]), "--sample_steps", str(config["steps"]),
                    "--frame_num", "81", "--sample_shift", "8",
                    "--sample_guide_scale", "6", "--sample_solver", "unipc",
                    "--offload_model", "False", "--save_file", str(pending)]
        sys.argv += config["method_args"][args.method]
        upstream_args = entry._parse_args()
        full_predictions = 0
        cold_start = pipeline is None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        print("GENERATING", args.method, row["id"], row["seed"], flush=True)
        entry.generate(upstream_args)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if pending.stat().st_size == 0:
            raise RuntimeError(f"Upstream entry saved an empty video: {pending}")
        pending.replace(video)
        result = dict(**row, method=args.method, seacache_thresh=float(config["method_args"]["seacache"][1]), steps=config["steps"], frames=81, size="832x480",
                      cfg=6, shift=8, solver="unipc", batch_size=1, nfe=full_predictions,
                      nfe_definition="Full backbone evaluations, counting both CFG branches",
                      wall_seconds_including_save=seconds,
                      latency=generation_seconds,
                      latency_definition="Official generation including text encoding and VAE decode; excludes model loading, saving and scoring",
                      cold_start_model_load_included=cold_start,
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      gpu=torch.cuda.get_device_name(), video=str(video))
        temporary = metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(metadata)
        pending_scores.append(scoring.submit(score_and_save, result, metadata))
        for future in pending_scores:
            if future.done():
                future.result()
        print("COMPLETED", args.method, row["id"], full_predictions, seconds, flush=True)
    for future in pending_scores:
        future.result()
    scoring.shutdown()
    scorer.close()


if __name__ == "__main__":
    main()
