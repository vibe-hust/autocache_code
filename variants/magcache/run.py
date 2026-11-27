"""MagCache adaptation for the two image backbones in Table 2."""
import ast
import contextlib
import io
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from diffusers import MagCacheConfig, QwenImage21Pipeline
from diffusers.hooks.mag_cache import (
    CacheContext, HookRegistry, TransformerBlockRegistry, apply_mag_cache,
    _MAG_CACHE_LEADER_BLOCK_HOOK,
)
from diffusers.hooks._helpers import TransformerBlockMetadata
from diffusers.models.transformers.transformer_flux3 import Flux3SingleTransformerBlock
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

from reactive_samplers.qwen_metrics import LegacyMetrics
from reactive_samplers.report import write_json
from reactive_samplers.context import Prediction


def read(path):
    return json.loads(Path(path).read_text())


def install(model, transformer, config):
    if model == 'flux3':
        TransformerBlockRegistry.register(Flux3SingleTransformerBlock,
            TransformerBlockMetadata(return_hidden_states_index=1,
                                     return_encoder_hidden_states_index=0))
        head = transformer.single_transformer_blocks[0]
    else:
        TransformerBlockRegistry.register(QwenImage21TransformerBlock,
            TransformerBlockMetadata())
        head = transformer.transformer_blocks[0]
    apply_mag_cache(transformer, config)
    events = []
    def observe(module, args, output):
        hook = HookRegistry.check_if_exists_or_initialize(head).get_hook(_MAG_CACHE_LEADER_BLOCK_HOOK)
        events.append(bool(hook.state_manager.get_state().should_compute))
    return head, events, head.register_forward_hook(observe)


def flux_image(adapter, row, steps):
    latent, sigmas = adapter.start_batch([row], steps)
    for sigma, target in zip(sigmas, sigmas[1:]):
        prediction = adapter.denoise(latent, sigma)
        latent = adapter.advance(latent, Prediction(prediction.guided.to(latent.dtype)), sigma, target)
    return adapter.finish_batch(latent)[0]


def summarize(rows):
    out = {'count': len(rows)}
    for name in ('nfe','seconds'):
        out['mean_'+name] = statistics.mean(r[name] for r in rows)
    for name in ('clip_score','image_reward','psnr','ssim','lpips'):
        out['mean_'+name] = statistics.mean(r['scores'][name] for r in rows)
    return out


def check_events(events, full_forwards):
    if len(events) != full_forwards or not any(not event for event in events):
        raise RuntimeError(f'MagCache inactive: {len(events)} forwards, {sum(events)} computed')


def load_generator(model, generation):
    steps = generation['reference_steps']
    if model == 'flux3':
        from variants.opal_search.flux3 import load_adapter
        adapter = load_adapter(generation, 'cuda:0')
        transformer = adapter.pipe.transformer
        def generate(row): return flux_image(adapter, row, steps)
        def sync(): adapter.synchronize()
    else:
        pipe = QwenImage21Pipeline.from_pretrained(generation['model_path'],
            torch_dtype=torch.bfloat16, local_files_only=True).to('cuda:0')
        pipe.vae.enable_slicing()
        transformer = pipe.transformer
        def generate(row):
            return pipe(prompt=row['prompt'], height=generation['height'],
                width=generation['width'], num_inference_steps=steps,
                true_cfg_scale=1., use_kv_cache=True,
                generator=torch.Generator(pipe._execution_device).manual_seed(row['seed'])).images[0]
        def sync(): torch.cuda.synchronize()
    return transformer, generate, sync


def calibrate(cfg, steps, transformer, generate):
    model = cfg['model']
    config = MagCacheConfig(calibrate=True, num_inference_steps=steps,
                            threshold=cfg['threshold'], max_skip_steps=cfg['max_skip_steps'],
                            retention_ratio=cfg['retention_ratio'])
    head, events, monitor = install(model, transformer, config)
    if model == 'flux3':
        branch = [0]
        def set_branch(module, args):
            hook = HookRegistry.check_if_exists_or_initialize(head).get_hook(_MAG_CACHE_LEADER_BLOCK_HOOK)
            hook.state_manager.set_context(CacheContext(name=f'branch{branch[0] % 2}'))
            branch[0] += 1
        transformer.register_forward_pre_hook(set_branch)
    calibration = io.StringIO()
    with contextlib.redirect_stdout(calibration):
        generate(read(cfg['calibration_prompt']))
    ratios = [ast.literal_eval(s) for s in re.findall(
        r'Copy these values to MagCacheConfig\(mag_ratios=\.\.\.\):\s*(\[[^\]]+\])', calibration.getvalue())]
    if not ratios or len(ratios[0]) != steps:
        raise RuntimeError(f'MagCache calibration did not produce {steps} ratios: {calibration.getvalue()[-1000:]}')
    monitor.remove()
    config = MagCacheConfig(mag_ratios=ratios[0], num_inference_steps=steps,
                            threshold=cfg['threshold'], max_skip_steps=cfg['max_skip_steps'],
                            retention_ratio=cfg['retention_ratio'])
    _, events, _ = install(model, transformer, config)
    return ratios, events


@torch.inference_mode()
def main():
    root = Path(sys.argv[1])
    cfg = read(root/'config.json')
    model = cfg['model']
    generation = read(cfg['generation'])
    rows = ([json.loads(s) for s in Path(cfg['prompts']).read_text().splitlines()]
            if cfg['prompts'].endswith('.jsonl') else read(cfg['prompts']))
    if len(rows) != 170:
        raise ValueError(f'Expected 170 DrawBench prompts, got {len(rows)}')
    (root/'records').mkdir(exist_ok=True)
    (root/'images').mkdir(exist_ok=True)
    steps = generation['reference_steps']
    references = {r['id']: r['image'] for r in read(cfg['reference_results'])['images']}
    full_forwards = steps * (2 if model == 'flux3' else 1)
    write_json(root/'status.json', {'stage':'loading','count':len(rows)})
    transformer, generate, sync = load_generator(model, generation)
    write_json(root/'status.json', {'stage':'calibrating'})
    ratios, events = calibrate(cfg, steps, transformer, generate)
    write_json(root/'calibration.json', {'ratios':ratios,'prompt':cfg['calibration_prompt']})
    metrics = LegacyMetrics(generation['metrics_python'], generation['metrics'])
    completed = []
    try:
        for row in rows:
            result = root/'records'/(row['id']+'.json')
            if result.exists():
                completed.append(read(result))
                continue
            write_json(root/'status.json', {'stage':'evaluating','completed':len(completed),'total':len(rows)})
            events.clear()
            sync(); start = time.perf_counter()
            image = generate(row)
            sync(); seconds = time.perf_counter()-start
            check_events(events, full_forwards)
            path = root/'images'/(row['id']+'.png')
            image.save(path)
            reference = references.get(row['id'])
            if reference is None or not Path(reference).exists():
                raise FileNotFoundError(f'Reference missing for {row["id"]}')
            record = {**row,'image':str(path),'reference':str(reference),
                      'nfe':sum(events),'seconds':seconds,'scores':metrics.score(str(path),row['prompt'],str(reference))}
            write_json(result,record)
            completed.append(record)
            write_json(root/'results.json',{'images':completed,'summary':summarize(completed)})
    finally:
        metrics.close()
    write_json(root/'status.json', {'stage':'completed','completed':len(completed)})

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(Path(sys.argv[1])/'status.json', {'stage':'failed','error':str(exc)})
        raise
