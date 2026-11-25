from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageChops
from reactive_samplers.qwen_metrics import LegacyMetrics
from .teacache_forward import teacache_forward


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


root = Path(sys.argv[1])
cfg = json.loads((root / 'config.json').read_text())
save(root / 'status.json', {'status': 'loading_model', 'threshold': cfg['threshold'], 'total': len(cfg['rows'])})
pipe = DiffusionPipeline.from_pretrained(cfg['model'], torch_dtype=torch.bfloat16,
                                       local_files_only=True).to('cuda')
pipe.set_progress_bar_config(disable=True)
model = pipe.transformer
model.forward = teacache_forward.__get__(model, type(model))
model.num_steps, model.rel_l1_thresh = cfg['steps'], cfg['threshold']
counter = {'nfe': 0}
def count_full(module, args):
    counter['nfe'] += 1
hook = model.single_transformer_blocks[-1].attn.register_forward_pre_hook(count_full)


def reset(enabled=True):
    model.enable_teacache, model.cnt = enabled, 0
    model.accumulated_rel_l1_distance = 0
    model.previous_modulated_input = model.previous_residual = None
    counter['nfe'] = 0


def generate(row):
    with torch.inference_mode(), redirect_stdout(io.StringIO()):
        embeds = pipe.encode_prompt(prompt=row['prompt'], prompt_2=None,
            device='cuda', num_images_per_prompt=1, max_sequence_length=512)[:2]
        return pipe(prompt=None, prompt_embeds=embeds[0], pooled_prompt_embeds=embeds[1],
            num_inference_steps=cfg['steps'], height=cfg['height'], width=cfg['width'],
            guidance_scale=cfg['guidance_scale'],
            generator=torch.Generator('cpu').manual_seed(row['seed'])).images[0]


metrics = LegacyMetrics(cfg['metrics_python'], {'names': ['psnr', 'ssim', 'lpips']})
partial = root / 'results.partial.json'
records = json.loads(partial.read_text()) if partial.exists() else []
done = {r['id'] for r in records}
try:
    for row in cfg['rows']:
        if row['id'] in done:
            continue
        reference = Path(cfg['reference_dir']) / (row['id'] + '.png')
        if not (root / 'control.json').exists():
            reset(False)
            control = generate(row)
            assert ImageChops.difference(control.convert('RGB'), Image.open(reference).convert('RGB')).getbbox() is None, 'TeaCache disabled output differs from reference'
            save(root / 'control.json', {'sample': row['id'], 'pixel_identical': True})
            reset()
            generate(row)
            print('CONTROL AND WARMUP PASSED', flush=True)
        reset()
        torch.cuda.synchronize()
        began = time.perf_counter()
        image = generate(row)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        path = root / 'images' / (row['id'] + '.png')
        image.save(path)
        scores = metrics.score(path, row['prompt'], reference)
        records.append({**row, 'method': 'teacache_030', 'image': str(path),
                        'nfe': counter['nfe'], 'seconds': seconds, 'scores': scores})
        save(partial, records)
        save(root / 'status.json', {'status': 'generating', 'completed': len(records), 'total': len(cfg['rows'])})
        print(row['id'], 'NFE', counter['nfe'], 'seconds', seconds, 'scores', scores,
              'completed', len(records), flush=True)
finally:
    metrics.close()
    hook.remove()
summary = {'count': len(records), 'mean_nfe': statistics.mean(r['nfe'] for r in records),
           'mean_seconds': statistics.mean(r['seconds'] for r in records)}
for key in ['psnr', 'ssim', 'lpips']:
    summary['mean_' + key] = statistics.mean(r['scores'][key] for r in records)
parent = json.loads(Path(cfg['reference_results']).read_text())
summary['speedup'] = parent['summary']['original']['mean_seconds'] / summary['mean_seconds']
save(root / 'generation_results.json', {'summary': {'teacache_030': summary}, 'images': records})
print('GENERATION COMPLETE', json.dumps(summary), flush=True)
