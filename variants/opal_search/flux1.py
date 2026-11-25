"""FLUX.1-dev: the same CPU noise, sigma grid and Euler arithmetic as BudCache."""
import copy
import time

import torch

from reactive_samplers.context import Context, Prediction, validate_skip
from reactive_samplers.models.sd3 import SD3Adapter
from reactive_samplers.recovery import policy_deadline, require_finite
from baselines.BudCache.sampling import get_noise, get_schedule, prepare, unpack, vae_decode


def load_adapter(generation, device):
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(generation['model_path'], torch_dtype=torch.bfloat16,
                                       local_files_only=True).to(device)
    adapter_type = BatchAdapter
    if generation.get('sampling_profile') == 'ecad':
        from variants.flux1_comparison.run import SearchAdapter
        adapter_type = SearchAdapter
    else:
        pipe.vae.to(dtype=torch.float32)
    return adapter_type(pipe, **{key: generation[key] for key in
                                ('guidance_scale', 'height', 'width', 'max_sequence_length')})


class BatchAdapter(SD3Adapter):
    model_kind = 'flux'

    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length):
        self.torch, self.pipe, self.device = torch, pipe, torch.device(pipe._execution_device)
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        self.max_sequence_length = max_sequence_length
        for model in (pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae):
            model.eval().requires_grad_(False)

    def start_batch(self, rows, steps):
        if len(rows) != 1:
            raise ValueError('This experiment uses one image per batch')
        row = rows[0]
        noise = get_noise(1, self.height, self.width, self.device, torch.bfloat16, row['seed'])
        self.inputs = prepare(noise, [row['prompt']], self.guidance_scale, self.pipe.text_encoder,
                              self.pipe.text_encoder_2, self.pipe.tokenizer, self.pipe.tokenizer_2)
        sigmas = get_schedule(steps, image_seq_len=self.inputs['img'].shape[1])
        self.reference_sigmas = torch.tensor(sigmas)
        self.reference_timesteps = self.reference_sigmas[:-1]*1000
        self.denoiser_evaluations = self.denoiser_items = 0
        return self.inputs['img'], tuple(sigmas)

    def view(self, indices, prefill=False):
        if indices != [0]:
            raise ValueError('Only the configured single-image batch is supported')
        return copy.copy(self)

    def timestep(self, sigma):
        return 1000*sigma

    def model_timestep(self, latent, sigma):
        return torch.full((latent.shape[0],), sigma, device=latent.device, dtype=latent.dtype)

    def prepare_feature(self, latent, sigma):
        return self.pipe.transformer.x_embedder(latent).detach()

    def denoise(self, latent, sigma):
        inputs = self.inputs
        output = self.pipe.transformer(hidden_states=latent, timestep=self.model_timestep(latent, sigma),
            guidance=inputs['guidance'], pooled_projections=inputs['pooled_prompt_embeds'],
            encoder_hidden_states=inputs['prompt_embeds'], txt_ids=inputs['text_ids'],
            img_ids=inputs['img_ids'], return_dict=False)[0]
        return Prediction(output)

    def refresh_residual(self, latent, sigma, feature):
        captured = []

        def capture(module, inputs):
            captured.append((inputs[0].detach()-feature).detach())

        with self.pipe.transformer.norm_out.register_forward_pre_hook(capture):
            prediction = self.denoise(latent, sigma)
        return prediction, captured[0]

    def reuse_residual(self, latent, sigma, feature, residual):
        model = self.pipe.transformer
        hidden = feature+residual
        timestep = self.model_timestep(latent, sigma)*1000
        guidance = self.inputs['guidance'].to(hidden.dtype)*1000
        temb = model.time_text_embed(timestep, guidance, self.inputs['pooled_prompt_embeds'])
        return Prediction(model.proj_out(model.norm_out(hidden, temb)))

    def advance(self, latent, prediction, sigma, next_sigma):
        return latent+(next_sigma-sigma)*prediction.guided

    def finish_batch(self, latent):
        return vae_decode(unpack(latent, self.height, self.width), self.pipe.vae, self.pipe.image_processor)


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
                    'consecutive_skips': ages[i], 'has_reference': residual is not None}, model='flux')
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
                residual = torch.empty_like(delta)
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
                nfe[i] += 1  # One guidance-distilled backbone call; reuse costs zero full calls.
                reference_sigma[i] = sigma
            traces[i].append({'step': step, 'sigma': sigma, 'next_sigma': target,
                'reuse_cache': reuse[i], 'nfe': nfe[i], 'observation': observations[i],
                'version': version, **policy_traces[i]})
        previous_latent, previous_feature, latent = latent, feature, updated
    adapter.synchronize()
    seconds = time.perf_counter() - began
    return latent, traces, {'batch_sampling_seconds': seconds, 'nfe': nfe,
        'backbone_batch_calls': adapter.denoiser_evaluations, 'backbone_items': adapter.denoiser_items}
