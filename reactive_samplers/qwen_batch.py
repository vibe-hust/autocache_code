"""Independent frozen-policy Qwen21 batching; existing single-image entry is untouched."""
import copy
import time

import numpy as np
import torch

from reactive_samplers.context import Context, Prediction, validate_skip
from reactive_samplers.models.qwen21 import Qwen21Adapter
from reactive_samplers.recovery import policy_deadline, require_finite


def select_kv(cache, indices):
    """Prefix KV is immutable after prefill; select the same rows as the images."""
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
    selected = QwenImage21KVCache(len(cache.layer_caches))
    for source, target in zip(cache.layer_caches, selected.layer_caches):
        key, value = source.get()
        target.store(key[indices], value[indices])
    return selected


class BatchAdapter(Qwen21Adapter):
    def start_batch(self, rows, steps):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        pipe = self.pipe
        self.embeds, self.embeds_mask, mask = pipe.encode_prompt(
            image=None, prompt=[r['prompt'] for r in rows], device=self.device, num_images_per_prompt=1)
        latent, _ = pipe.prepare_latents(None, len(rows), pipe.transformer.config.in_channels,
            self.height, self.width, self.embeds.dtype, self.device,
            [torch.Generator(device=self.device).manual_seed(r['seed']) for r in rows], None)
        self.image_mask = torch.cat([mask, mask.new_ones(mask.shape[0], latent.shape[1] // 4)], dim=1)
        self.image_shapes = [[(1, self.height // pipe.vae_scale_factor,
                                self.width // pipe.vae_scale_factor)] for _ in rows]
        cfg = pipe.scheduler.config
        mu = calculate_shift(latent.shape[1], cfg.get('base_image_seq_len', 256),
            cfg.get('max_image_seq_len', 4096), cfg.get('base_shift', .5), cfg.get('max_shift', 1.15))
        retrieve_timesteps(pipe.scheduler, steps, self.device, sigmas=np.linspace(1., 1 / steps, steps), mu=mu)
        pipe.scheduler.set_begin_index(0)
        self.reference_sigmas = pipe.scheduler.sigmas.clone()
        self.reference_timesteps = pipe.scheduler.timesteps.clone()
        sigmas = tuple(map(float, self.reference_sigmas))
        self._timestep_by_sigma = dict(zip(sigmas[:-1], self.reference_timesteps))
        self.kv_cache = QwenImage21KVCache(len(pipe.transformer.transformer_blocks))
        self.denoiser_items = self.denoiser_evaluations = 0
        return latent, sigmas

    def view(self, indices, prefill=False):
        # A shallow adapter view shares weights, never mutates parent conditioning.
        view = copy.copy(self)
        view.embeds = self.embeds[indices]
        view.embeds_mask = None if self.embeds_mask is None else self.embeds_mask[indices]
        view.image_mask = self.image_mask[indices]
        view.image_shapes = [self.image_shapes[i] for i in indices]
        view.kv_cache = self.kv_cache if prefill else select_kv(self.kv_cache, indices)
        view.denoiser_evaluations = 0 if prefill else 1
        return view

    def finish_batch(self, latent):
        # Reuse the existing decoder exactly; decode one image at a time to bound memory.
        return [self.finish(item[None]) for item in latent]


@torch.inference_mode()
def sample_batch(adapter, rows, programs, steps, max_consecutive_skips=4, timeout=30, on_step=None):
    if not rows or steps < 1 or max_consecutive_skips < 0:
        raise ValueError('Require samples, positive steps and nonnegative reuse limit')
    latent, sigmas = adapter.start_batch(rows, steps)
    count = len(rows)
    memory = [{} for _ in rows]
    nfe, ages = [0] * count, [0] * count
    reference_sigma = [None] * count
    previous_latent = previous_feature = residual = None
    traces = [[] for _ in rows]
    version = programs.version
    adapter.synchronize()
    began = time.perf_counter()
    for step, (sigma, target) in enumerate(zip(sigmas, sigmas[1:])):
        if programs.version != version:
            raise RuntimeError('Batch policy must remain frozen')
        feature = adapter.prepare_feature(latent, sigma)
        require_finite(feature, 'batch input feature')
        reuse, observations, policy_traces = [], [], []
        for i, row in enumerate(rows):
            ctx = Context(row['prompt'], row['seed'], step, nfe[i], steps, sigma, sigmas[-1],
                adapter.timestep(sigma), sigmas, latent[i:i+1], Prediction(None), [],
                sample_id=row['id'], method='residual_cache', input_feature=feature[i:i+1],
                reference_feature=None if previous_feature is None else previous_feature[i:i+1],
                reference_latent=None if previous_latent is None else previous_latent[i:i+1],
                cache_observation={'sigma': sigma, 'step_delta_sigma': sigma-target,
                    'since_real_delta_sigma': 0. if reference_sigma[i] is None else reference_sigma[i]-sigma,
                    'consecutive_skips': ages[i], 'has_reference': residual is not None}, model='qwen21')
            with policy_deadline(timeout):
                observation = programs.analysis.analyze(ctx, memory[i])
                action = programs.updater.choose(observation, ctx, memory[i])
                requested = validate_skip(action)
            reuse.append(requested and residual is not None and ages[i] < max_consecutive_skips)
            observations.append(observation)
            forced = 'initialization' if residual is None else 'reuse_limit' if ages[i] >= max_consecutive_skips else None
            policy_traces.append({'requested_skip': requested, 'forced': forced,
                                  'reason': copy.deepcopy(programs.trace)})
        refresh = [i for i in range(count) if not reuse[i]]
        cached = [i for i in range(count) if reuse[i]]
        prediction = torch.empty_like(latent)
        if refresh:
            view = adapter.view(refresh, prefill=step == 0)
            fresh, delta = view.refresh_residual(latent[refresh], sigma, feature[refresh])
            prediction[refresh] = fresh.guided
            if residual is None:
                residual = torch.empty_like(feature)
            residual[refresh] = delta
            adapter.denoiser_evaluations += 1
            adapter.denoiser_items += len(refresh)
        if cached:
            view = copy.copy(adapter)
            # Reuse runs only the output head and does not access prefix KV.
            prediction[cached] = view.reuse_residual(latent[cached], sigma,
                feature[cached], residual[cached]).guided
        require_finite(prediction, 'batch prediction')
        if on_step:
            on_step(step, latent, prediction, reuse)
        updated = adapter.advance(latent, Prediction(prediction), sigma, target)
        require_finite(updated, 'batch latent')
        for i in range(count):
            ages[i] = ages[i] + 1 if reuse[i] else 0
            if not reuse[i]:
                nfe[i] += 1
                reference_sigma[i] = sigma
            traces[i].append({'step': step, 'sigma': sigma, 'next_sigma': target,
                'reuse_cache': reuse[i], 'nfe': nfe[i], 'observation': observations[i],
                'version': version, **policy_traces[i]})
        previous_latent, previous_feature, latent = latent, feature, updated
    adapter.synchronize()
    seconds = time.perf_counter() - began
    return latent, traces, {'batch_sampling_seconds': seconds, 'nfe': nfe,
        'backbone_batch_calls': adapter.denoiser_evaluations, 'backbone_items': adapter.denoiser_items}
