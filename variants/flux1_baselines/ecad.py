import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def worker(root, gpu):
    import torch
    from ecad.image_generators.flux_image_generator import FluxImageGenerator
    from ecad.schedulers.cache_scheduler.flux_cache_schedule import FluxCacheSchedule
    from reactive_samplers.qwen_metrics import LegacyMetrics

    cfg = json.loads((root / 'config.json').read_text())
    generator = FluxImageGenerator(weights_name=cfg['model'])
    generator.pipeline_weights = cfg['model']
    generator.create_encoder_pipeline()
    generator.create_diffusion_pipeline()
    generator.diffusion_pipeline.set_progress_bar_config(disable=True)
    transformer = generator.diffusion_pipeline.transformer
    schedules = {'original': generator.cache_schedule}
    for name, path in cfg['schedules'].items():
        schedules[name] = FluxCacheSchedule.from_json(Path(path))
    metrics = LegacyMetrics(cfg['metrics_python'], {'names': ['psnr', 'ssim', 'lpips']})
    shard = root / f'gpu{gpu}.json'
    records = json.loads(shard.read_text())['images'] if shard.exists() else []
    done = {(r['method'], r['id']) for r in records}
    print(f'GPU{gpu} MODEL READY; steps={generator.num_inference_steps}', flush=True)
    try:
        for row in cfg['rows'][gpu::2]:
            if all((name, row['id']) in done for name in schedules):
                continue
            generator.start_seed = row['seed']
            reference = root / 'original' / (row['id'] + '.png')
            for name, schedule in schedules.items():
                if (name, row['id']) in done:
                    continue
                generator.cache_schedule = schedule
                generator.callbacks[1] = schedule.per_step_callback
                transformer.cache_schedule = schedule
                for block in list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks):
                    block.cache_schedule = schedule
                schedule.reset_step()
                transformer.reset_cache()
                torch.cuda.synchronize()
                began = time.perf_counter()
                embed = generator.encode_prompts([row['prompt']], batch_size=1)[0]
                embed = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in embed.items()}
                image = generator.generate_images(embed, images_per_prompt=1,
                    height=256, width=256, guidance_scale=5)[0][0]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                output = root / name / (row['id'] + '.png')
                image.save(output)
                result = {**row, 'method': name, 'image': str(output), 'seconds': seconds}
                if name != 'original':
                    result['scores'] = metrics.score(output, row['prompt'], reference)
                records.append(result)
                done.add((name, row['id']))
                save(shard, {'images': records})
                print(f'GPU{gpu} {name} {row["id"]} seconds={seconds:.3f} scores={result.get("scores")}', flush=True)
    finally:
        metrics.close()


def main(root):
    jobs = []
    for gpu in (0, 1):
        log = (root / f'gpu{gpu}.log').open('a')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        proc = subprocess.Popen([sys.executable, '-u', __file__, str(root), '--gpu', str(gpu)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((proc, log))
    codes = []
    for proc, log in jobs:
        codes.append(proc.wait())
        log.close()
    save(root / 'exit_codes.json', codes)
    if any(codes):
        raise SystemExit('Worker failed; see GPU logs')
    rows = [r for gpu in (0, 1) for r in json.loads((root / f'gpu{gpu}.json').read_text())['images']]
    summary = {}
    for method in ('original', 'ecad_fast', 'ecad_faster'):
        selected = [r for r in rows if r['method'] == method]
        summary[method] = {'count': len(selected), 'mean_seconds': statistics.mean(r['seconds'] for r in selected)}
        if method != 'original':
            for key in ('psnr', 'ssim', 'lpips'):
                summary[method]['mean_' + key] = statistics.mean(r['scores'][key] for r in selected)
    save(root / 'results.json', {'summary': summary, 'images': rows})
    save(root / 'status.json', {'status': 'completed', 'summary': summary})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args.root) if args.gpu is None else worker(args.root, args.gpu)
