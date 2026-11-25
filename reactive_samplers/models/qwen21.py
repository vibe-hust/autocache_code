"""Qwen-Image-2.1 T2I: official prefix KV cache plus image-only residual cache."""
import numpy as np

from .flux import FluxAdapter
from ..context import Prediction


class Qwen21Adapter(FluxAdapter):
    model_kind = "qwen21"

    def __init__(self, pipe, *, guidance_scale, height, width, max_sequence_length):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        if type(pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
            raise TypeError("Qwen-Image-2.1 requires the official FlowMatch Euler scheduler")
        if guidance_scale != 1 or not pipe.transformer.config.causal_condition:
            raise ValueError("Qwen21 T2I uses official guidance 1 and prefix KV caching")
        if height % 32 or width % 32 or pipe.config.sample_sigmas is not None:
            raise ValueError("Use the non-Turbo model and image dimensions divisible by 32")
        self.torch, self.pipe, self.device = torch, pipe, pipe._execution_device
        self.guidance_scale, self.height, self.width = guidance_scale, height, width
        self.max_sequence_length, self.cfg = max_sequence_length, False
        for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
            model.requires_grad_(False)
            model.eval()

    def start(self, prompt, seed, reference_steps):
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
        torch, pipe = self.torch, self.pipe
        with torch.inference_mode():
            self.embeds, self.embeds_mask, mask = pipe.encode_prompt(
                image=None, prompt=prompt, device=self.device, num_images_per_prompt=1)
            latent, _ = pipe.prepare_latents(
                None, 1, pipe.transformer.config.in_channels, self.height, self.width,
                self.embeds.dtype, self.device, torch.Generator(device=self.device).manual_seed(seed), None)
        self.image_mask = torch.cat([mask, mask.new_ones(mask.shape[0], latent.shape[1] // 4)], dim=1)
        self.image_shapes = [[(1, self.height // pipe.vae_scale_factor, self.width // pipe.vae_scale_factor)]]
        config = pipe.scheduler.config
        mu = calculate_shift(latent.shape[1], config.get("base_image_seq_len", 256),
                             config.get("max_image_seq_len", 4096), config.get("base_shift", .5),
                             config.get("max_shift", 1.15))
        retrieve_timesteps(pipe.scheduler, reference_steps, self.device,
                           sigmas=np.linspace(1., 1 / reference_steps, reference_steps), mu=mu)
        pipe.scheduler.set_begin_index(0)
        self.reference_sigmas = pipe.scheduler.sigmas.clone()
        self.reference_timesteps = pipe.scheduler.timesteps.clone()
        sigmas = tuple(map(float, self.reference_sigmas))
        self._timestep_by_sigma = dict(zip(sigmas[:-1], self.reference_timesteps))
        self.kv_cache = QwenImage21KVCache(len(pipe.transformer.transformer_blocks))
        self.denoiser_items = self.denoiser_evaluations = 0
        self.synchronize()
        return latent, sigmas

    def predict(self, latent, sigma, feature_layers):
        if feature_layers:
            raise ValueError("Qwen21 cache decisions use pre-prediction image features only")
        with self.torch.inference_mode(), self.pipe.transformer.cache_context("cond"):
            output = self.pipe.transformer(
                hidden_states=latent, timestep=self.model_timestep(latent, sigma),
                encoder_hidden_states=self.embeds, encoder_hidden_states_mask=self.embeds_mask,
                img_shapes=self.image_shapes, img_mask=self.image_mask, attention_kwargs={},
                kv_cache=self.kv_cache,
                kv_cache_mode="extract" if self.denoiser_evaluations == 0 else "cached",
                return_dict=False)[0]
            self.denoiser_evaluations += 1
            self.denoiser_items += latent.shape[0]
            return Prediction(output[:, -latent.shape[1]:])

    def prepare_feature(self, latent, sigma):
        with self.torch.inference_mode():
            return self.pipe.transformer.img_in(latent).detach()

    def refresh_residual(self, latent, sigma, feature):
        captured = []
        def capture(module, inputs):
            # Prefill includes text; subsequent forwards contain only target tokens.
            hidden = inputs[0][:, -feature.shape[1]:]
            if hidden.shape != feature.shape:
                raise ValueError("Qwen21 residual must contain target-image tokens only")
            captured.append((hidden.detach() - feature).detach())
        hook = self.pipe.transformer.norm_out.register_forward_pre_hook(capture)
        try:
            prediction = self.predict(latent, sigma, [])
        finally:
            hook.remove()
        if len(captured) != 1:
            raise ValueError("Expected one Qwen21 output boundary")
        return prediction, captured[0]

    def reuse_residual(self, latent, sigma, feature, residual):
        torch, model = self.torch, self.pipe.transformer
        with torch.inference_mode():
            hidden = feature + residual
            t = self.model_timestep(latent, sigma).to(hidden.dtype)
            t = torch.cat([t, t.new_zeros(1)], dim=0)
            temb = model.time_text_embed(t, hidden)
            target_mask = torch.ones(hidden.shape[1], device=hidden.device, dtype=torch.bool)
            return Prediction(model.proj_out(model.norm_out(hidden, temb, target_mask)))

    def finish(self, latent):
        torch, pipe = self.torch, self.pipe
        with torch.inference_mode():
            z = pipe._unpack_latents(latent, self.height, self.width, pipe.vae_scale_factor).to(pipe.vae.dtype)
            mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(z)
            std = torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(z)
            image = pipe.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
            return pipe.image_processor.postprocess(image, output_type="pil")[0]


def verify_pipeline(pipe, adapter, prompt, seed, steps):
    """Gate experiments on exact all-refresh parity, including prefix caching."""
    latent, sigmas = adapter.start(prompt, seed, steps)
    for sigma, target in zip(sigmas, sigmas[1:]):
        feature = adapter.prepare_feature(latent, sigma)
        prediction, _ = adapter.refresh_residual(latent, sigma, feature)
        latent = adapter.advance(latent, prediction, sigma, target)
    expected = pipe(prompt=prompt, num_inference_steps=steps, true_cfg_scale=1.,
                    height=adapter.height, width=adapter.width, use_kv_cache=True,
                    generator=adapter.torch.Generator(device=adapter.device).manual_seed(seed),
                    output_type="latent").images
    adapter.torch.testing.assert_close(latent, expected, rtol=0, atol=0)
    return {"status": "passed", "max_abs_latent_error": float((latent - expected).abs().max()),
            "steps": steps, "seed": seed, "rtol": 0, "atol": 0, "use_kv_cache": True}
