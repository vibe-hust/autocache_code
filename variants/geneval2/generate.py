"""Generate GenEval 2 with the frozen DrawBench policy or paired Qwen baselines."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['official', 'frozen', 'seacache', 'dicache'])
    parser.add_argument('--config', default='experiments/geneval2/generation.json', type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    rows = json.loads(Path(cfg['prompts']).read_text())
    rank, workers = int(os.environ.get('SLURM_PROCID',0)), int(os.environ.get('SLURM_NTASKS',1))
    import torch
    from diffusers import QwenImage21Pipeline
    from baselines.qwen21_cache.qwen21_batch_cache import Qwen21BatchCache
    from variants.seacache.qwen21 import Qwen21SeaCache
    from variants.opal_search.policy import Runtime
    from reactive_samplers.qwen_batch import BatchAdapter, sample_batch
    from reactive_samplers.report import write_json

    pipe = QwenImage21Pipeline.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
                                              local_files_only=True).to('cuda:0')
    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.enable_slicing()
    adapter = BatchAdapter(pipe, guidance_scale=1., height=cfg['height'], width=cfg['width'], max_sequence_length=None)
    folder = Path(cfg['output_dir'])/args.method
    folder.mkdir(parents=True,exist_ok=True)
    grid = (cfg['height']//pipe.vae_scale_factor,cfg['width']//pipe.vae_scale_factor)
    steps = cfg['reference_steps']
    for row in rows[rank::workers]:
        path = folder/(row['id']+'.png')
        if path.exists() and path.with_suffix('.json').exists(): continue
        if args.method == 'frozen':
            runtime = Runtime(cfg['policy'])
            controller = nullcontext()
        elif args.method == 'seacache':
            controller = Qwen21SeaCache(pipe.transformer,pipe.scheduler,steps,cfg['baseline_thresholds']['seacache'],grid)
        elif args.method == 'dicache':
            controller = Qwen21BatchCache(pipe.transformer,'dicache',steps,cfg['baseline_thresholds']['dicache'])
        else:
            controller = nullcontext()
        torch.cuda.synchronize(); began = time.perf_counter()
        with torch.inference_mode(), controller:
            if args.method == 'frozen':
                latent, trace, timing = sample_batch(adapter,[row],runtime,steps,max_consecutive_skips=steps)
                image = adapter.finish_batch(latent)[0]
                nfe = timing['nfe'][0]
            else:
                image = pipe(prompt=row['prompt'],height=cfg['height'],width=cfg['width'],
                    num_inference_steps=steps,true_cfg_scale=1.,use_kv_cache=True,
                    generator=torch.Generator(device='cuda').manual_seed(row['seed'])).images[0]
                nfe = steps if args.method == 'official' else controller.counts[0]
        torch.cuda.synchronize(); seconds = time.perf_counter()-began
        image.save(path)
        write_json(path.with_suffix('.json'),{**row,'method':args.method,'nfe':nfe,'seconds':seconds})


if __name__ == '__main__': main()
