import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace


class RandomPrograms:
    version = 'random-reuse-p070-v1'

    def __init__(self, seed, action):
        self.rng, self.action = random.Random(seed), action
        self.analysis = SimpleNamespace(analyze=lambda ctx, memory: dict(ctx.cache_observation))
        self.updater = SimpleNamespace(choose=self.choose)
        self.trace = {}

    def choose(self, observation, ctx, memory):
        draw = self.rng.random() if observation['has_reference'] else None
        reuse = draw is not None and draw < 0.7
        self.trace = {'uniform': draw, 'reuse_probability': 0.7}
        return self.action(reuse)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def worker(root, rank):
    import torch
    from diffusers import QwenImage21Pipeline
    from reactive_samplers.context import SkipAction
    from reactive_samplers.qwen_batch import BatchAdapter, sample_batch
    from reactive_samplers.qwen_metrics import LegacyMetrics

    cfg = json.loads((root / 'config.json').read_text())
    g = cfg['generation']
    references = json.loads(Path(cfg['reference_results']).read_text())
    rows = references['context']['rows']
    refs = {row['id']: row['image'] for row in references['images']}
    shard = root / f'gpu{rank}.json'
    records = json.loads(shard.read_text())['images'] if shard.exists() else []
    completed = {row['id'] for row in records}
    pipe = QwenImage21Pipeline.from_pretrained(
        g['model_path'], torch_dtype=getattr(torch, g['dtype']), local_files_only=True).to('cuda:0')
    pipe.transformer.eval().requires_grad_(False)
    adapter = BatchAdapter(pipe, **{key: g[key] for key in
        ('guidance_scale', 'height', 'width', 'max_sequence_length')})
    metrics = LegacyMetrics(g['metrics_python'], g['metrics'])
    try:
        for index in range(rank, len(rows), 2):
            row = rows[index]
            if row['id'] in completed:
                continue
            policy_seed = cfg['policy_seed'] + index
            programs = RandomPrograms(policy_seed, SkipAction)
            with torch.inference_mode():
                adapter.synchronize()
                began = time.perf_counter()
                latent, traces, timing = sample_batch(adapter, [row], programs,
                    g['reference_steps'], max_consecutive_skips=g['reference_steps'])
                image = adapter.finish_batch(latent)[0]
                adapter.synchronize()
                seconds = time.perf_counter() - began
            image_path = root / 'images' / (row['id'] + '.png')
            trace_path = root / 'trajectories' / (row['id'] + '.jsonl')
            image.save(image_path)
            trace_path.write_text(''.join(json.dumps(event) + '\n' for event in traces[0]))
            record = {**row, 'image': str(image_path), 'trajectory': str(trace_path),
                      'policy_seed': policy_seed, 'nfe': timing['nfe'][0], 'seconds': seconds}
            print(f"GPU{rank} generated {row['id']} NFE={record['nfe']} seconds={seconds:.3f}", flush=True)
            record['scores'] = metrics.score(image_path, row['prompt'], refs[row['id']])
            records.append(record)
            write(shard, {'images': records})
            print(f"GPU{rank} scored {row['id']} completed={len(records)}", flush=True)
    finally:
        metrics.close()


def main(root):
    jobs = []
    for gpu in (0, 1):
        log = (root / f'gpu{gpu}.log').open('a')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'TOKENIZERS_PARALLELISM': 'false'}
        process = subprocess.Popen([sys.executable, '-u', '-m', 'variants.random_reuse.run', str(root), '--gpu', str(gpu)],
                                   env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((process, log))
    codes = []
    for process, log in jobs:
        codes.append(process.wait())
        log.close()
    write(root / 'exit_codes.json', codes)
    if any(codes):
        raise SystemExit('Worker failed; see gpu0.log and gpu1.log')
    rows = sorted([row for gpu in (0, 1) for row in
        json.loads((root / f'gpu{gpu}.json').read_text())['images']], key=lambda row: row['id'])
    summary = {'count': len(rows), 'mean_nfe': statistics.mean(row['nfe'] for row in rows),
               'mean_seconds': statistics.mean(row['seconds'] for row in rows)}
    for key in ('clip_score', 'image_reward', 'psnr', 'ssim', 'lpips'):
        values = [row['scores'][key] for row in rows]
        summary['mean_' + key] = statistics.mean(values) if all(v is not None for v in values) else None
    write(root / 'results.json', {'summary': summary, 'images': rows})
    write(root / 'status.json', {'status': 'completed', **summary})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.gpu is None:
        cfg = json.loads(args.config.read_text())
        root = Path(cfg.pop('output_dir'))
        root.mkdir(parents=True, exist_ok=False)
        for name in ('images', 'trajectories'):
            (root/name).mkdir()
        write(root/'config.json', cfg)
        main(root)
    else:
        worker(args.config, args.gpu)
