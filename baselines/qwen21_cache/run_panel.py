"""Queue 40-step Qwen baselines after the Wan jobs; one image per GPU at a time."""
import argparse
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from reactive_samplers.report import write_json
from reactive_samplers.data import drawbench_splits


METHODS = ('original', 'seacache', 'dicache', 'teacache')
METRICS = ('clip_score', 'image_reward', 'psnr', 'ssim', 'lpips')


def assigned(rows, gpu, gpus=(0, 1)):
    return rows[gpus.index(gpu)::len(gpus)]


def summary(rows):
    result = {'count': len(rows), 'mean_nfe': statistics.mean(r['nfe'] for r in rows),
              'mean_seconds': statistics.mean(r['seconds'] for r in rows)}
    for name in METRICS:
        values = [r['scores'][name] for r in rows]
        result['mean_' + name] = None if name == 'psnr' and None in values else statistics.mean(values)
    return result


def load_pipeline(cfg):
    import torch
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(
        cfg['model_path'], torch_dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    return pipe


def cache_for(pipe, cfg, method):
    if method == 'original':
        return None
    steps, threshold = cfg['reference_steps'], cfg['thresholds'][method]
    if method == 'seacache':
        from variants.seacache.qwen21 import Qwen21SeaCache

        grid = (cfg['height'] // pipe.vae_scale_factor, cfg['width'] // pipe.vae_scale_factor)
        return Qwen21SeaCache(pipe.transformer, pipe.scheduler, steps, threshold, grid)
    from baselines.qwen21_cache.qwen21_cache import Qwen21Cache

    return Qwen21Cache(pipe.transformer, method, steps, threshold)


def generate_image(pipe, cfg, row, callback=None):
    import torch

    return pipe(prompt=row['prompt'], height=cfg['height'], width=cfg['width'],
                num_inference_steps=cfg['reference_steps'], true_cfg_scale=cfg['guidance_scale'],
                use_kv_cache=True,
                generator=torch.Generator(pipe._execution_device).manual_seed(row['seed']),
                callback_on_step_end=callback).images[0]


def worker(cfg, gpu):
    import torch
    from reactive_samplers.qwen_metrics import LegacyMetrics

    root = Path(cfg['output_dir'])
    rows = assigned(json.loads(Path(cfg['prompts']).read_text()), gpu, cfg.get('gpus', [0, 1]))
    shard = root / f'gpu{gpu}' / 'results.json'
    records = json.loads(shard.read_text())['images'] if shard.exists() else []
    done = {(r['method'], r['id']) for r in records}
    steps = cfg['reference_steps']
    methods = cfg.get('methods', METHODS)
    pending = any((method, row['id']) not in done for method in methods for row in rows)
    if pending:
        print(f'GPU{gpu}: loading Qwen; {len(rows)} prompts, {steps} steps', flush=True)
        pipe = load_pipeline(cfg)
        for method in methods:
            for row in rows:
                if (method, row['id']) in done:
                    continue
                cache = cache_for(pipe, cfg, method)
                events = []

                def record(pipeline, index, timestep, tensors):
                    node = {'step': index, 'timestep': float(timestep),
                            'sigma': float(pipeline.scheduler.sigmas[index])}
                    if cache is None:
                        events.append({**node, 'reuse_cache': False, 'nfe': index + 1})
                    elif method != 'seacache':
                        cache.events[-1].update(node)
                    return tensors

                print(f'GPU{gpu}: {method} {row["id"]}', flush=True)
                torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.inference_mode(), (cache if cache is not None else nullcontext()):
                    image = generate_image(pipe, cfg, row, record)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - began
                folder = root / method
                (folder / 'images').mkdir(parents=True, exist_ok=True)
                (folder / 'trajectories').mkdir(exist_ok=True)
                image_path = folder / 'images' / (row['id'] + '.png')
                trace_path = folder / 'trajectories' / (row['id'] + '.jsonl')
                image.save(image_path)
                if cache is not None:
                    events = cache.events[0] if method == 'seacache' else cache.events
                trace_path.write_text(''.join(json.dumps(e) + '\n' for e in events))
                nfe = steps if cache is None else cache.counts[0] if method == 'seacache' else cache.nfe
                records.append({**row, 'method': method, 'image': str(image_path),
                                'trajectory': str(trace_path), 'gpu': gpu, 'steps': steps,
                                'nfe': nfe, 'seconds': elapsed})
                write_json(shard, {'images': records})
        del cache, pipe, image
        gc.collect()
        torch.cuda.empty_cache()
    metrics = LegacyMetrics(cfg['metrics_python'], cfg['metrics'])
    try:
        for row in records:
            if 'scores' in row:
                continue
            reference = Path(cfg.get('reference_dir', root / 'original' / 'images')) / (row['id'] + '.png')
            row['scores'] = metrics.score(row['image'], row['prompt'], reference)
            write_json(shard, {'images': records})
            print(f'GPU{gpu}: scored {row["method"]} {row["id"]}', flush=True)
    finally:
        metrics.close()


def queue(cfg, config_path):
    root = Path(cfg['output_dir'])
    gpus = cfg.get('gpus', [0, 1])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'config.json', cfg)
    jobs = []
    for gpu in gpus:
        with (root / f'gpu{gpu}.log').open('a') as log:
            jobs.append(subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()), str(config_path), '--worker', str(gpu)],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'].split(',')[gpu] if os.environ.get('CUDA_VISIBLE_DEVICES') else str(gpu)}, stdout=log, stderr=subprocess.STDOUT))
    codes = [job.wait() for job in jobs]
    if any(codes):
        raise RuntimeError(f'Qwen workers failed: {codes}; see gpu0.log / gpu1.log')
    comparison = aggregate(cfg)
    print(json.dumps(comparison, indent=2), flush=True)


def aggregate(cfg):
    root = Path(cfg['output_dir'])
    records = [row for gpu in cfg.get('gpus', [0, 1])
               for row in json.loads((root / f'gpu{gpu}' / 'results.json').read_text())['images']]
    comparison = {}
    prompt_dir = Path(__file__).resolve().parents[2] / 'experiments/qwen21-drawbench30-170'
    for method in cfg.get('methods', METHODS):
        images = [r for r in records if r['method'] == method]
        comparison[method] = summary(images)
        comparison[method]['splits'] = {
            name: summary(group) for name, group in drawbench_splits(images, prompt_dir).items()}
        write_json(root / method / 'results.json', {'images': images, 'summary': comparison[method]})
    write_json(root / 'comparison.json', comparison)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--worker', type=int, metavar='GPU_INDEX')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.worker is None:
        queue(cfg, args.config.resolve())
    else:
        worker(cfg, args.worker)


if __name__ == '__main__':
    main()
