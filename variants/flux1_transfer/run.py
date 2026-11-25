import json
from pathlib import Path
import statistics
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def reset(model, method, cfg, scheduler):
    model.enable_dicache = method == 'dicache'
    model.enable_seacache = method == 'seacache'
    model.cnt, model.num_steps = 0, cfg['steps']
    model.accumulated_rel_l1_distance = 0
    model.previous_modulated_input = model.previous_residual = None
    model.previous_input = model.previous_probe_states = None
    model.previous_probe_residual = model.previous_output = None
    model.residual_window, model.probe_residual_window = [], []
    model.probe_depth, model.ret_ratio, model.error_choice = 1, 0.2, 'delta_y'
    model.rel_l1_thresh = cfg['thresholds']['dicache']
    model.seacache_thresh = cfg['thresholds']['seacache']
    model.scheduler = scheduler


def main(root):
    import torch
    from PIL import Image, ImageChops
    from ecad.image_generators.flux_image_generator import FluxImageGenerator
    from reactive_samplers.context import SkipAction
    from reactive_samplers.qwen_metrics import LegacyMetrics
    from .whole_backbone import WholeBackboneCache
    from .dicache_compat import dicache_forward
    from .seacache_compat import seacache_forward

    cfg = json.loads((root / 'config.json').read_text())
    policy_name = cfg['policy_name']
    print(f'LOADING MODEL; steps={cfg["steps"]}; prompts={len(cfg["rows"])}; methods=original,{policy_name},dicache,seacache', flush=True)
    namespace = {'SkipAction': SkipAction}
    for name in ['signals.py', 'updater.py']:
        exec(compile((root / 'policy' / name).read_text(), name, 'exec'), namespace)
    FluxImageGenerator.DEFAULT_NUM_INFERENCE_STEPS = cfg['steps']
    generator = FluxImageGenerator(weights_name=cfg['model'])
    generator.pipeline_weights = cfg['model']
    generator.create_encoder_pipeline()
    generator.create_diffusion_pipeline()
    generator.diffusion_pipeline.set_progress_bar_config(disable=True)
    model = generator.diffusion_pipeline.transformer
    original = model.forward
    policy = WholeBackboneCache(generator, namespace['extract_signals'], namespace['choose'])
    forwards = {'original': original, policy_name: policy.forward,
                'dicache': dicache_forward.__get__(model, type(model)),
                'seacache': seacache_forward.__get__(model, type(model))}
    counter = {'nfe': 0}
    def count_full(module, args):
        counter['nfe'] += 1
    hook = model.single_transformer_blocks[-1].register_forward_pre_hook(count_full)
    metrics = LegacyMetrics(cfg['metrics_python'], {'names': ['psnr', 'ssim', 'lpips']})
    partial = root / 'results.partial.json'
    records = json.loads(partial.read_text()) if partial.exists() else []
    completed = {(r['method'], r['id']) for r in records}
    controls_path = root / 'controls.json'
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else {}
    print('MODEL READY', flush=True)
    try:
        for row in cfg['rows']:
            if all((name, row['id']) in completed for name in forwards):
                continue
            generator.start_seed = row['seed']
            reference = root / 'original' / (row['id'] + '.png')
            for method, forward in forwards.items():
                if (method, row['id']) in completed:
                    continue
                reset(model, method, cfg, generator.diffusion_pipeline.scheduler)
                if method in ('dicache', 'seacache') and method not in controls:
                    model.forward = forward
                    setattr(model, 'enable_' + method, False)
                    embed = generator.encode_prompts([row['prompt']], batch_size=1)[0]
                    embed = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in embed.items()}
                    control = generator.generate_images(embed, height=256, width=256, guidance_scale=5)[0][0]
                    assert ImageChops.difference(control.convert('RGB'), Image.open(reference).convert('RGB')).getbbox() is None, method + ' disabled-cache output differs from original'
                    controls[method] = {'sample': row['id'], 'pixel_identical': True}
                    save(controls_path, controls)
                    print('DISABLED-CACHE CONTROL PASSED', method, flush=True)
                    reset(model, method, cfg, generator.diffusion_pipeline.scheduler)
                policy.reset(row)
                counter['nfe'] = 0
                events = []
                def trace_forward(**kwargs):
                    before = counter['nfe']
                    result = forward(**kwargs)
                    events.append({'step': len(events), 'reuse_cache': before == counter['nfe'],
                                   'nfe': counter['nfe']})
                    return result
                model.forward = trace_forward
                torch.cuda.synchronize()
                began = time.perf_counter()
                embed = generator.encode_prompts([row['prompt']], batch_size=1)[0]
                embed = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in embed.items()}
                image = generator.generate_images(embed, height=256, width=256, guidance_scale=5)[0][0]
                torch.cuda.synchronize()
                seconds = time.perf_counter() - began
                image_path = root / method / (row['id'] + '.png')
                image.save(image_path)
                trace_path = root / 'trajectories' / method / (row['id'] + '.jsonl')
                trace_path.write_text(''.join(json.dumps(e) + '\n' for e in (policy.events if method == policy_name else events)))
                record = {**row, 'method': method, 'image': str(image_path), 'trajectory': str(trace_path),
                          'nfe': counter['nfe'], 'seconds': seconds}
                if method != 'original':
                    record['scores'] = metrics.score(image_path, row['prompt'], reference)
                records.append(record)
                completed.add((method, row['id']))
                save(partial, records)
                save(root / 'status.json', {'status': 'generating', 'completed': len(records),
                    'total': len(cfg['rows']) * len(forwards), 'last_method': method, 'last_sample': row['id']})
                print(f'{method} {row["id"]} NFE={counter["nfe"]} seconds={seconds:.4f} scores={record.get("scores")} completed={len(records)}/{len(cfg["rows"])*len(forwards)}', flush=True)
    finally:
        metrics.close()
        hook.remove()
    summary = {}
    for method in forwards:
        group = [r for r in records if r['method'] == method]
        summary[method] = {'count': len(group), 'mean_nfe': statistics.mean(r['nfe'] for r in group),
                           'mean_seconds': statistics.mean(r['seconds'] for r in group)}
        if method != 'original':
            for key in ['psnr', 'ssim', 'lpips']:
                summary[method]['mean_' + key] = statistics.mean(r['scores'][key] for r in group)
    for method in summary:
        summary[method]['speedup'] = summary['original']['mean_seconds'] / summary[method]['mean_seconds']
    save(root / 'generation_results.json', {'summary': summary, 'images': records})
    print('GENERATION COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
