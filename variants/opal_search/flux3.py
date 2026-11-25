"""FLUX3 Klein Base adapter for the unchanged OPAL search contract.

Official blocks own text encoding, latent initialization, CFG, Euler updates and
decoding. Each node has one policy decision and two independent CFG residuals.
"""
import copy
from functools import wraps
import time
from unittest.mock import patch

import torch

from reactive_samplers.context import Context, Prediction, validate_skip
from reactive_samplers.models.flux import FluxAdapter
from reactive_samplers.recovery import policy_deadline, require_finite


def load_adapter(generation, device):
    from diffusers import ModularPipeline
    checkpoint = generation['model_path']
    blocks = ModularPipeline.from_pretrained(checkpoint, local_files_only=True).blocks
    blocks.sub_blocks.pop('prompt_upsampler')  # Fixed baseline captions are supplied by the dataset.
    pipe = blocks.init_pipeline(checkpoint)
    pipe.load_components(names=['transformer', 'text_encoder', 'tokenizer', 'vae', 'scheduler'],
                         dtype={'vae': torch.float32, 'default': getattr(torch, generation['dtype'])})
    pipe.to(device)
    pipe.guider = pipe.guider.new(guidance_scale=generation['guidance_scale'])
    return BatchAdapter(pipe, **{k: generation[k] for k in
                                ('guidance_scale', 'height', 'width', 'max_sequence_length')})


class BatchAdapter(FluxAdapter):
    model_kind = 'flux3'

    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length):
        if pipe.transformer.config.flow_map or guidance_scale != 4:
            raise ValueError('This experiment requires Klein Base with CFG=4')
        self.torch, self.pipe, self.device = torch, pipe, torch.device(pipe._execution_device)
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        self.max_sequence_length = max_sequence_length
        self.core = pipe.blocks.sub_blocks['denoise'].sub_blocks['text2image']
        for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
            model.eval().requires_grad_(False)

    def start_batch(self, rows, steps):
        from diffusers.modular_pipelines.modular_pipeline import PipelineState
        if len(rows) != 1:
            raise ValueError('Match the Qwen search configuration: batch_size=1')
        row = rows[0]
        self.state = PipelineState(values={
            'prompt': row['prompt'], 'height': self.height, 'width': self.width,
            'max_sequence_length': self.max_sequence_length, 'num_inference_steps': steps,
            'generator': torch.Generator(device=self.device).manual_seed(row['seed'])})
        self.pipe.blocks.sub_blocks['text_encoder'](self.pipe, self.state)
        for name in ('input', 'prepare_latents', 'set_timesteps', 'prepare_rope_inputs'):
            self.core.sub_blocks[name](self.pipe, self.state)
        self.reference_sigmas = self.pipe.scheduler.sigmas.clone()
        self.reference_timesteps = self.pipe.scheduler.timesteps.clone()
        sigmas = tuple(map(float, self.reference_sigmas))
        self._timestep_by_sigma = dict(zip(sigmas[:-1], self.reference_timesteps))
        self.denoiser_evaluations = self.denoiser_items = 0
        return self.state.get('latents'), sigmas

    def view(self, indices, prefill=False):
        if indices != [0]:
            raise ValueError('Only the configured single-image batch is supported')
        return copy.copy(self)

    def prepare_feature(self, latent, sigma):
        return self.pipe.transformer.x_embedders['image'](latent).detach()

    def denoise(self, latent, sigma):
        # Delegate both branch order and guidance combination to the official block.
        self.state.set('latents', latent)
        step = self.pipe.scheduler.step_index
        if step is None:
            step = 0
        denoiser = self.core.sub_blocks['denoise'].sub_blocks['denoiser']
        block_state = denoiser.get_block_state(self.state)
        block_state.num_inference_steps = len(self.reference_timesteps)
        _, block_state = denoiser(self.pipe, block_state, step, self._timestep_by_sigma[sigma])
        return Prediction(block_state.noise_pred)

    def refresh_residual(self, latent, sigma, feature):
        captured = []
        def capture(module, inputs):
            captured.append((inputs[0].detach() - feature).detach())
        with self.pipe.transformer.norm_out.register_forward_pre_hook(capture):
            prediction = self.denoise(latent, sigma)
        if len(captured) != 2:
            raise ValueError('Expected two separate official CFG branches')
        return prediction, torch.stack(captured, dim=1)

    def reuse_residual(self, latent, sigma, feature, residual):
        model = self.pipe.transformer
        branches = []
        @wraps(model.forward)  # The official denoiser inspects this signature.
        def head(**kwargs):
            branch = len(branches)
            branches.append(branch)
            hidden = feature + residual[:, branch]
            t = kwargs['timestep'].to(kwargs['hidden_states'].dtype) * 1000
            temb = model.timestep_embedder(model.time_proj(t).to(hidden.dtype))[:, None]
            temb = temb.expand(-1, hidden.shape[1], -1)
            return (model.proj_out(model.norm_out(hidden, temb)),)
        with patch.object(model, 'forward', head):
            prediction = self.denoise(latent, sigma)
        if len(branches) != 2:
            raise ValueError('Expected two separate official CFG branches')
        return prediction

    def finish_batch(self, latent):
        self.state.set('latents', latent)
        self.core.sub_blocks['after_denoise'](self.pipe, self.state)
        self.pipe.blocks.sub_blocks['decode'](self.pipe, self.state)
        return self.state.get('images')




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
            ctx = Context(row['prompt'], row['seed'], step, nfe[i], 2 * steps, sigma, sigmas[-1],
                adapter.timestep(sigma), sigmas, latent[i:i+1], Prediction(None), [],
                sample_id=row['id'], method='residual_cache', input_feature=feature[i:i+1],
                reference_feature=None if previous_feature is None else previous_feature[i:i+1],
                reference_latent=None if previous_latent is None else previous_latent[i:i+1],
                cache_observation={'sigma': sigma, 'step_delta_sigma': sigma-target,
                    'since_real_delta_sigma': 0. if reference_sigma[i] is None else reference_sigma[i]-sigma,
                    'consecutive_skips': ages[i], 'has_reference': residual is not None}, model='flux3')
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
                nfe[i] += 2  # Both full CFG branch calls; reuse costs zero full calls.
                reference_sigma[i] = sigma
            traces[i].append({'step': step, 'sigma': sigma, 'next_sigma': target,
                'reuse_cache': reuse[i], 'nfe': nfe[i], 'observation': observations[i],
                'version': version, **policy_traces[i]})
        previous_latent, previous_feature, latent = latent, feature, updated
    adapter.synchronize()
    seconds = time.perf_counter() - began
    return latent, traces, {'batch_sampling_seconds': seconds, 'nfe': nfe,
        'backbone_batch_calls': 2 * adapter.denoiser_evaluations, 'backbone_items': 2 * adapter.denoiser_items}
