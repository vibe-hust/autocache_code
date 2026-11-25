"""Official BudCache Stage 1 on raw DrawBench, paired with full computation."""
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from diffusers import FluxPipeline

from baselines.BudCache.cache_manager import reset2original, set_budcache
from baselines.BudCache.sampling import denoising_euler, get_noise, get_schedule, prepare, unpack, vae_decode
from reactive_samplers.qwen_metrics import LegacyMetrics
from reactive_samplers.report import write_json


def summarize(records):
    summaries = {}
    for split in ('exploration', 'evaluation', 'all'):
        rows = [r for r in records if split == 'all' or r['split'] == split]
        if not rows:
            continue
        methods = {}
        for method in ('official', 'budcache'):
            means = {'count': len(rows)}
            for key in ('nfe', 'seconds'):
                means['mean_'+key] = statistics.mean(r[method][key] for r in rows)
            for key in ('psnr', 'ssim', 'lpips', 'clip_score', 'image_reward'):
                values = [r[method]['scores'][key] for r in rows]
                means['mean_'+key] = None if None in values else statistics.mean(values)
            methods[method] = means
        methods['official']['speedup'] = 1.0
        methods['budcache']['speedup'] = methods['official']['mean_seconds']/methods['budcache']['mean_seconds']
        summaries[split] = methods
    return summaries


def save_results(root, records):
    write_json(root/'results.json', {'images': records, 'summaries': summarize(records),
               'psnr_null': 'positive infinity for identical images',
               'latency_scope': 'prompt encoding, seeded noise, cache control, denoising and VAE decode; excludes loading, warmup, PNG saving and scoring',
               'speedup_formula': 'mean matched full-computation latency / mean BudCache latency'})


@torch.inference_mode()
def generate(pipe, row, cfg, cache_steps, counter):
    counter[0] = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    if cache_steps is None:
        reset2original(pipe.transformer)
    else:
        set_budcache(pipe.transformer, cache_steps, cfg['steps'])
    noise = get_noise(1, cfg['height'], cfg['width'], torch.device('cuda:0'), torch.bfloat16, row['seed'])
    inputs = prepare(noise, [row['prompt']], cfg['guidance_scale'], pipe.text_encoder,
                     pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2)
    sigmas = get_schedule(cfg['steps'], image_seq_len=inputs['img'].shape[1])
    latent = denoising_euler(pipe.transformer, inputs['img'], sigmas, inputs['guidance'],
                            inputs['pooled_prompt_embeds'], inputs['prompt_embeds'],
                            inputs['text_ids'], inputs['img_ids'])
    image = vae_decode(unpack(latent, cfg['height'], cfg['width']), pipe.vae, pipe.image_processor)[0]
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    expected = cfg['steps'] if cache_steps is None else cfg['steps']-len(cache_steps)
    if counter[0] != expected:
        raise RuntimeError(f'Expected {expected} backbone evaluations, observed {counter[0]}')
    return image, seconds, counter[0]


def main():
    root = Path(sys.argv[1])
    cfg = json.loads((root/'config.json').read_text())
    rows = json.loads((root/'prompts.json').read_text())
    cache_steps = json.loads(Path(cfg['schedule']).read_text())['cache_step']
    assert cfg['steps'] == 28 and len(set(cache_steps)) == 18 and 0 not in cache_steps
    for name in ('official', 'budcache', 'records'):
        (root/name).mkdir(exist_ok=True)
    write_json(root/'status.json', {'stage': 'loading', 'total': len(rows)})
    pipe = FluxPipeline.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
                                       local_files_only=True).to('cuda:0')
    pipe.vae.to(dtype=torch.float32)
    counter = [0]

    def count(module, args):
        counter[0] += 1

    handle = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(count)
    write_json(root/'status.json', {'stage': 'warmup', 'total': len(rows)})
    generate(pipe, rows[0], cfg, None, counter)
    generate(pipe, rows[0], cfg, cache_steps, counter)
    metrics = LegacyMetrics(cfg['metrics_python'], cfg['metrics'])
    records = []
    try:
        for row in rows:
            record_path = root/'records'/(row['id']+'.json')
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            write_json(root/'status.json', {'stage': 'evaluating', 'completed': len(records),
                                          'total': len(rows), 'current': row['id']})
            record = dict(row)
            for method, steps in [('official', None), ('budcache', cache_steps)]:
                image, seconds, nfe = generate(pipe, row, cfg, steps, counter)
                path = root/method/(row['id']+'.png')
                image.save(path)
                reference = root/'official'/(row['id']+'.png')
                record[method] = {'image': str(path), 'nfe': nfe, 'seconds': seconds,
                                  'scores': metrics.score(path, row['prompt'], reference)}
            record['speedup'] = record['official']['seconds']/record['budcache']['seconds']
            write_json(record_path, record)
            records.append(record)
            save_results(root, records)
            print(f"{len(records)}/{len(rows)} {row['id']} NFE={record['budcache']['nfe']} latency={record['budcache']['seconds']:.3f}s", flush=True)
    finally:
        metrics.close()
        handle.remove()
        reset2original(pipe.transformer)
    save_results(root, records)
    write_json(root/'status.json', {'stage': 'completed', 'completed': len(records), 'total': len(rows)})


if __name__ == '__main__':
    main()
