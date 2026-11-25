"""Evaluate frozen r09_s03 at the released ECAD Fast generation settings."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import calculate_shift

from reactive_samplers.qwen_metrics import LegacyMetrics
from variants.opal_search.flux1 import BatchAdapter, sample_batch
from variants.opal_search.policy import Runtime


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


class ECADSettingsAdapter(BatchAdapter):
    """Keep AutoCache residual reuse; match ECAD's stock Diffusers inputs/Euler."""

    def prepare_prompt(self, row):
        self.embeds = self.pipe.encode_prompt(
            prompt=[row['prompt']], prompt_2=None, device=self.device,
            num_images_per_prompt=1, max_sequence_length=512)

    def start_batch(self, rows, steps):
        prompt, pooled, text_ids = self.embeds
        latent, image_ids = self.pipe.prepare_latents(
            1, self.pipe.transformer.config.in_channels // 4, self.height, self.width,
            prompt.dtype, self.device, torch.Generator('cpu').manual_seed(rows[0]['seed']), None)
        scheduler = self.pipe.scheduler
        c = scheduler.config
        mu = calculate_shift(latent.shape[1], c.base_image_seq_len, c.max_image_seq_len,
                             c.base_shift, c.max_shift)
        scheduler.set_timesteps(steps, device=self.device,
                                sigmas=np.linspace(1., 1. / steps, steps), mu=mu)
        self.reference_sigmas = scheduler.sigmas
        self.reference_timesteps = scheduler.timesteps
        sigmas = tuple(float(s) for s in scheduler.sigmas)
        self.sigma_indices = {s: i for i, s in enumerate(sigmas[:-1])}
        self.inputs = dict(prompt_embeds=prompt, pooled_prompt_embeds=pooled,
                           text_ids=text_ids, img_ids=image_ids,
                           guidance=torch.tensor([self.guidance_scale], device=self.device))
        self.denoiser_evaluations = self.denoiser_items = 0
        return latent, sigmas

    def model_timestep(self, latent, sigma):
        t = self.reference_timesteps[self.sigma_indices[sigma]]
        return t.expand(latent.shape[0]).to(latent.dtype) / 1000

    def advance(self, latent, prediction, sigma, next_sigma):
        t = self.reference_timesteps[self.sigma_indices[sigma]]
        return self.pipe.scheduler.step(prediction.guided, t, latent, return_dict=False)[0]

    def finish_batch(self, latent):
        pipe = self.pipe
        latent = pipe._unpack_latents(latent, self.height, self.width, pipe.vae_scale_factor)
        latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        return pipe.image_processor.postprocess(pipe.vae.decode(latent, return_dict=False)[0])


class SearchAdapter(ECADSettingsAdapter):
    """Encode each new prompt inside the shared search evaluator's timer."""

    def start_batch(self, rows, steps):
        self.prepare_prompt(rows[0])
        return super().start_batch(rows, steps)


def summary(records):
    return {'count': len(records), **{
        'mean_' + key: statistics.mean(r[key] for r in records)
        for key in ('nfe', 'seconds') if key in records[0]}, **{
        'mean_' + key: statistics.mean(r['scores'][key] for r in records)
        for key in records[0].get('scores', {}) if key != 'psnr' or records[0]['scores'][key] is not None}}


@torch.inference_mode()
def ours(root, pilot):
    cfg = json.loads((root / 'config.json').read_text())
    pipe = FluxPipeline.from_pretrained(cfg['model'], torch_dtype=torch.bfloat16,
                                        local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    adapter = ECADSettingsAdapter(pipe, guidance_scale=5, height=256, width=256,
                                  max_sequence_length=512)
    rows = cfg['rows'][:3] if pilot else cfg['rows']
    output = root / ('pilot' if pilot else 'ours')
    output.mkdir(exist_ok=True)
    policy = Path(cfg['policy'])

    def generate(row, name):
        torch.cuda.synchronize()
        began = time.perf_counter()
        adapter.prepare_prompt(row)
        trace = None
        if name == 'original':
            image = pipe(prompt_embeds=adapter.embeds[0], pooled_prompt_embeds=adapter.embeds[1],
                         height=256, width=256, num_inference_steps=20, guidance_scale=5,
                         generator=torch.Generator('cpu').manual_seed(row['seed'])).images[0]
            nfe = 20
        else:
            runtime = Runtime(root / 'all_new' if name == 'all_new' else policy)
            latent, traces, timing = sample_batch(adapter, [row], runtime, 20, max_consecutive_skips=20)
            image = adapter.finish_batch(latent)[0]
            trace, nfe = traces[0], timing['nfe'][0]
        torch.cuda.synchronize()
        return image, nfe, time.perf_counter() - began, trace

    generate(rows[0], 'original')
    generate(rows[0], 'r09_s03')  # Warm up both paths; exclude these timings.
    metrics = LegacyMetrics(cfg['metrics_python'], cfg['metrics'])
    records = []
    try:
        for row in rows:
            reference = output / (row['id'] + '-original.png')
            original, nfe, seconds, _ = generate(row, 'original')
            original.save(reference)
            records.append(dict(row, method='original', image=str(reference), nfe=nfe, seconds=seconds))
            if pilot:
                all_new, _, _, _ = generate(row, 'all_new')
                assert np.array_equal(np.asarray(original), np.asarray(all_new)), row['id']
            image, nfe, seconds, trace = generate(row, 'r09_s03')
            path = output / (row['id'] + '-r09_s03.png')
            image.save(path)
            (output / (row['id'] + '-trajectory.jsonl')).write_text(
                ''.join(json.dumps(t, allow_nan=False) + '\n' for t in trace))
            records.append(dict(row, method='r09_s03', image=str(path), nfe=nfe, seconds=seconds,
                                scores=metrics.score(path, row['prompt'], reference)))
            groups = {name: summary([r for r in records if r['method'] == name])
                      for name in ('original', 'r09_s03')}
            groups['r09_s03']['speedup'] = groups['original']['mean_seconds'] / groups['r09_s03']['mean_seconds']
            save(output / 'results.json', {'summary': groups, 'images': records})
            save(output / 'status.json', {'status': 'running', 'completed': len(records) // 2, 'total': len(rows)})
            print(json.dumps({'id': row['id'], 'nfe': nfe, 'seconds': seconds,
                              'scores': records[-1]['scores'], 'summary': groups}), flush=True)
    finally:
        metrics.close()
    save(output / 'status.json', {'status': 'completed', 'completed': len(rows), 'total': len(rows),
                                 'all_new_pixel_parity': 'passed' if pilot else 'verified in pilot'})


def ecad(root):
    """Reuse the aligned 170 completed samples; generate only the missing 30."""
    from ecad.image_generators.flux_image_generator import FluxImageGenerator
    from ecad.schedulers.cache_scheduler.flux_cache_schedule import FluxCacheSchedule

    cfg = json.loads((root / 'config.json').read_text())
    output = root / 'ecad'
    output.mkdir(exist_ok=True)
    saved = output / 'results.json'
    records = json.loads(saved.read_text())['images'] if saved.exists() else []
    done = {r['id'] for r in records}
    FluxImageGenerator.DEFAULT_NUM_INFERENCE_STEPS = cfg['steps']
    generator = FluxImageGenerator(weights_name=cfg['model'])
    generator.pipeline_weights = cfg['model']
    generator.create_encoder_pipeline()
    generator.create_diffusion_pipeline()
    generator.diffusion_pipeline.set_progress_bar_config(disable=True)
    transformer = generator.diffusion_pipeline.transformer
    schedules = {'original': generator.cache_schedule,
                 'ecad_fast': FluxCacheSchedule.from_json(Path(cfg['ecad_schedule']))}
    metrics = LegacyMetrics(cfg['metrics_python'], cfg['metrics'])
    try:
        for row in (r for r in cfg['rows'] if r['id'] not in done):
            generator.start_seed = row['seed']
            reference = output / (row['id'] + '-original.png')
            for name, schedule in schedules.items():
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
                path = output / (row['id'] + '-' + name + '.png')
                image.save(path)
                record = dict(row, method=name, image=str(path), seconds=seconds)
                record['scores'] = metrics.score(path, row['prompt'], reference)
                records.append(record)
            groups = {name: summary([r for r in records if r['method'] == name])
                      for name in ('original', 'ecad_fast')}
            groups['ecad_fast']['speedup'] = groups['original']['mean_seconds'] / groups['ecad_fast']['mean_seconds']
            save(output / 'results.json', {'summary': groups, 'images': records})
            save(output / 'status.json', {'status': 'running', 'completed': len(records) // 2, 'total': 200})
            print('ECAD', row['id'], len(records) // 2, flush=True)
    finally:
        metrics.close()
    save(output / 'status.json', {'status': 'completed', 'completed': 200, 'total': 200})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('mode', choices=['pilot', 'ours', 'ecad'])
    args = parser.parse_args()
    ecad(args.root) if args.mode == 'ecad' else ours(args.root, args.mode == 'pilot')
