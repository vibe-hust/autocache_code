"""FLUX.1-dev residual cache; official Diffusers 0.35.1 model and Euler updates."""
import numpy as np

from .sd3 import SD3Adapter
from ..context import Prediction


class FluxAdapter(SD3Adapter):
    # Only state, synchronization and serialization helpers are inherited.
    model_kind = "flux"

    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        if type(pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise TypeError("FLUX requires official FlowMatchEulerDiscreteScheduler")
        if pipe.scheduler.config.invert_sigmas or pipe.scheduler.config.get("stochastic_sampling", False):
            raise ValueError("FLUX residual cache requires deterministic descending Euler")
        if not pipe.transformer.config.guidance_embeds:
            raise ValueError("Only guidance-distilled FLUX.1-dev is supported")
        self.torch, self.pipe, self.device = torch, pipe, pipe._execution_device
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        self.max_sequence_length = max_sequence_length
        self.cfg = False  # Embedded guidance, not conditional/unconditional CFG duplication.
        for model in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
            if model is not None:
                model.requires_grad_(False)
                model.eval()

    def start(self, prompt, seed, reference_steps):
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        torch, pipe = self.torch, self.pipe
        with torch.inference_mode():
            self.embeds, self.pooled, self.text_ids = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, device=self.device,
                num_images_per_prompt=1, max_sequence_length=self.max_sequence_length)
            latent, self.image_ids = pipe.prepare_latents(
                1, pipe.transformer.config.in_channels // 4, self.height, self.width,
                self.embeds.dtype, self.device, torch.Generator(device=self.device).manual_seed(seed), None)
        sigmas = np.linspace(1.0, 1 / reference_steps, reference_steps)
        if pipe.scheduler.config.get("use_flow_sigmas", False):
            sigmas = None
        config = pipe.scheduler.config
        mu = calculate_shift(latent.shape[1], config.get("base_image_seq_len", 256),
                             config.get("max_image_seq_len", 4096), config.get("base_shift", 0.5),
                             config.get("max_shift", 1.15))
        retrieve_timesteps(pipe.scheduler, reference_steps, self.device, sigmas=sigmas, mu=mu)
        pipe.scheduler.set_begin_index(0)
        self.reference_sigmas = pipe.scheduler.sigmas.clone()
        self.guidance = torch.full([1], self.guidance_scale, device=self.device, dtype=torch.float32)
        self.denoiser_items = self.denoiser_evaluations = 0
        sigmas = tuple(map(float, self.reference_sigmas))
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError("FLUX scheduler must produce a descending sigma grid")
        self.reference_timesteps = pipe.scheduler.timesteps.clone()
        self._timestep_by_sigma = dict(zip(sigmas[:-1], self.reference_timesteps))
        self.synchronize()
        return latent, sigmas

    def advance(self, latent, prediction, sigma, next_sigma):
        # Every residual-cache node updates. Preserve the official full schedule,
        # including sigma device: moving its scalar to CPU changes BF16 rounding.
        with self.torch.inference_mode():
            return self.pipe.scheduler.step(
                prediction.guided, self._timestep_by_sigma[sigma], latent, return_dict=False)[0]

    def model_timestep(self, latent, sigma):
        # Use the actual official label, then cast/divide in the official order.
        return self._timestep_by_sigma[sigma].expand(latent.shape[0]).to(latent.dtype) / 1000

    def predict(self, latent, sigma, feature_layers):
        if feature_layers:
            raise ValueError("FLUX residual decisions expose only pre-prediction input features")
        with self.torch.inference_mode():
            self.denoiser_evaluations += 1
            self.denoiser_items += latent.shape[0]
            with self.pipe.transformer.cache_context("cond"):
                output = self.pipe.transformer(
                    hidden_states=latent, timestep=self.model_timestep(latent, sigma),
                    guidance=self.guidance, pooled_projections=self.pooled,
                    encoder_hidden_states=self.embeds, txt_ids=self.text_ids, img_ids=self.image_ids,
                    joint_attention_kwargs=None, return_dict=False)[0]
            return Prediction(output)

    def prepare_feature(self, latent, sigma):
        with self.torch.inference_mode():
            return self.pipe.transformer.x_embedder(latent).detach()

    def refresh_residual(self, latent, sigma, feature):
        captured = []
        def capture(module, inputs):
            if inputs[0].shape != feature.shape:
                raise ValueError("FLUX residual boundary must contain image tokens only")
            captured.append((inputs[0].detach() - feature).detach())
        hook = self.pipe.transformer.norm_out.register_forward_pre_hook(capture)
        try:
            # Keep the official forward intact, including its inexpensive x_embedder call.
            prediction = self.predict(latent, sigma, [])
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("Expected exactly one FLUX output boundary")
        return prediction, captured[0]

    def reuse_residual(self, latent, sigma, feature, residual):
        with self.torch.inference_mode():
            model = self.pipe.transformer
            hidden = feature + residual
            timestep = self.model_timestep(latent, sigma).to(hidden.dtype) * 1000
            guidance = self.guidance.to(hidden.dtype) * 1000
            temb = model.time_text_embed(timestep, guidance, self.pooled)
            return Prediction(model.proj_out(model.norm_out(hidden, temb)))

    def finish(self, latent):
        pipe = self.pipe
        with self.torch.inference_mode():
            z = pipe._unpack_latents(latent, self.height, self.width, pipe.vae_scale_factor)
            z = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            image = pipe.vae.decode(z, return_dict=False)[0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]


def verify_pipeline(pipe, adapter, prompt, seed, steps):
    """No-reuse adapter versus official FluxPipeline; no additional Agent calls."""
    latent, sigmas = adapter.start(prompt, seed, steps)
    for sigma, target in zip(sigmas, sigmas[1:]):
        latent = adapter.advance(latent, adapter.predict(latent, sigma, []), sigma, target)
    expected = pipe(prompt=prompt, prompt_2=prompt, num_inference_steps=steps,
                    guidance_scale=adapter.guidance_scale, true_cfg_scale=1.0,
                    height=adapter.height, width=adapter.width,
                    max_sequence_length=adapter.max_sequence_length,
                    generator=adapter.torch.Generator(device=adapter.device).manual_seed(seed),
                    output_type="latent").images
    adapter.torch.testing.assert_close(latent, expected, rtol=0, atol=0)
    return {"status": "passed", "max_abs_latent_error": float((latent - expected).abs().max()),
            "steps": steps, "seed": seed, "rtol": 0, "atol": 0}
