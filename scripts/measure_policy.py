from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch
from reactive_samplers.operator_counter import OperatorCounter


case = Path(sys.argv[1])
counting = sys.argv[2] == "count"
config = json.loads((case / "config.json").read_text())
generation, model = config["generation"], config["model"]
output = case / ("operator" if counting else "normal")
output.mkdir(exist_ok=True)
if model == "wan":
    from variants.wan21_opal.evaluate import Evaluator
    evaluator = Evaluator(generation, 1, 64 * 1024 * 1024)
    evaluator.load()
elif model == "qwen":
    from diffusers import QwenImage21Pipeline
    from reactive_samplers.qwen_batch import BatchAdapter, sample_batch
    from variants.opal_search.policy import Runtime
    pipe = QwenImage21Pipeline.from_pretrained(
        generation["model_path"], torch_dtype=torch.bfloat16, local_files_only=True).to("cuda:0")
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    adapter = BatchAdapter(pipe, **{k: generation[k] for k in
                           ("guidance_scale", "height", "width", "max_sequence_length")})
else:
    from variants.opal_search.flux3 import load_adapter, sample_batch
    from variants.opal_search.policy import Runtime
    adapter = load_adapter(generation, "cuda:0")


def generate():
    if model == "wan":
        folder = output / row["id"]
        folder.mkdir(exist_ok=True)
        return evaluator.generate(row, case / "policy", folder, None), None
    with torch.inference_mode():
        latent, traces, timing = sample_batch(
            adapter, [row], runtime, generation["reference_steps"],
            max_consecutive_skips=generation["reference_steps"])
        image = adapter.finish_batch(latent)[0]
    return dict(id=row["id"], seed=row["seed"], nfe=timing["nfe"][0]), image


records = []
for row in config["samples"]:
    if model != "wan":
        runtime = Runtime(case / "policy", 64 * 1024 * 1024)
        if counting:
            runtime.extract.__builtins__["__import__"] = __import__
    torch.cuda.synchronize()
    began = time.perf_counter()
    if counting:
        meter = OperatorCounter()
        with meter.wan_attention(evaluator.pipe.model) if model == "wan" else nullcontext():
            record, image = meter.run(generate)
        record.update(meter.record())
    else:
        record, image = generate()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - began
    record["gpu"] = torch.cuda.get_device_name()
    if counting:
        record["seconds_with_counting"] = elapsed
    else:
        record.setdefault("seconds", elapsed)
    if image is not None and not counting:
        image.save(output / (row["id"] + ".png"))
    (output / (row["id"] + ".json")).write_text(json.dumps(record, indent=2))
    records.append(record)
    summary = dict(count=len(records), mean_nfe=statistics.mean(r["nfe"] for r in records))
    if counting:
        summary["mean_tflops"] = statistics.mean(r["tflops"] for r in records)
    else:
        summary["mean_seconds"] = statistics.mean(r["seconds"] for r in records)
    (case / ("operator_flops.json" if counting else "latency.json")).write_text(
        json.dumps(dict(samples=records, summary=summary), indent=2))
    print(model, config["strategy"], row["id"], summary, flush=True)
