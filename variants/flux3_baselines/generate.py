"""Generate with original DrawBench text; one independent Slurm task per GPU."""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers import ModularPipeline

from .cache import cache


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["official", "teacache", "dicache", "seacache"])
    parser.add_argument("checkpoint", help="Local official Diffusers Klein Base snapshot")
    parser.add_argument("output", type=Path, help="Shared root for the four methods")
    parser.add_argument("--threshold", type=float, help="Explicit operating point for each cache")
    args = parser.parse_args()
    cached = args.mode in ("teacache", "dicache", "seacache")
    if cached and (args.threshold is None or not math.isfinite(args.threshold) or args.threshold <= 0):
        parser.error("Choose a positive --threshold for this method")
    parser_rows = Path("experiments/flux3-drawbench30-170")
    rows = [json.loads(line) for name in ("development.jsonl", "test.jsonl")
            for line in (parser_rows/name).read_text().splitlines() if line.strip()]
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    setup = json.loads(Path("experiments/flux3-drawbench30-170/baselines.json").read_text())
    (root/"prompts.json").write_text(json.dumps(rows,ensure_ascii=False))
    checkpoint = str(Path(args.checkpoint).resolve(strict=True))
    pipe = ModularPipeline.from_pretrained(checkpoint, local_files_only=True)
    blocks = pipe.blocks
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline(checkpoint)
    pipe.load_components(names=["transformer", "text_encoder", "tokenizer", "vae", "scheduler"],
                         dtype={"vae": torch.float32, "default": torch.bfloat16})
    if pipe.transformer.config.flow_map:
        raise ValueError("Expected the Base checkpoint, got the distilled flow map")
    pipe.to("cuda")
    pipe.guider = pipe.guider.new(guidance_scale=4.0)

    def generate(caption, seed):
        return pipe(prompt=caption, height=1024, width=1024, num_inference_steps=50,
                    generator=torch.Generator(device="cuda").manual_seed(seed), output="images")[0]

    folder = root / args.mode
    folder.mkdir(exist_ok=True)
    rank = int(os.environ.get("SLURM_PROCID",0))
    workers = int(os.environ.get("SLURM_NTASKS",1))
    for row in rows[rank::workers]:
        caption = row["prompt"]
        path = folder / (row["id"] + ".png")
        record = path.with_suffix(".json")
        if record.exists() and path.exists():
            if json.loads(record.read_text())["threshold"] != args.threshold:
                raise ValueError("Choose another output root for a different threshold")
            continue
        controller = cache(pipe.transformer, pipe.scheduler, args.mode, args.threshold,
                           setup["teacache_coefficients"]) if cached else nullcontext(([], []))
        with controller as (events, _):
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = generate(caption, row["seed"])
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
        image.save(path)
        write(record, dict(**row, caption=caption, prompt_mode="original", method=args.mode, steps=50, cfg=4.,
                           threshold=args.threshold, coefficients=setup["teacache_coefficients"] if args.mode=="teacache" else None,
                           ret_ratio=.2 if args.mode=="dicache" else None, probe_depth=1 if args.mode=="dicache" else None,
                           full_cfg_branch_calls=sum(not e["reuse"] for e in events)
                           if cached else 100, seconds=seconds, decisions=events))
        print("COMPLETED", args.mode, row["id"], flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
