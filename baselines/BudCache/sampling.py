# Adapted from Black Forest Labs FLUX (https://github.com/black-forest-labs/flux),
# licensed under the Apache License, Version 2.0.
# Modifications Copyright 2026 BudCache Authors.
import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor



def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)

def prepare(img: Tensor, prompt: str | list[str], guidance, clip_encoder, t5_encoder, clip_tokenizer, t5_tokenizer) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    device = img.device
    weight_dtype = img.dtype
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> (h w) c",)
    # img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    clip_tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length,
                                      truncation=True, return_overflowing_tokens=False,
                                      return_length=False, return_tensors="pt").input_ids.to(device)
    pooled_prompt_embeds = clip_encoder(clip_tokens, output_hidden_states=False).pooler_output.to(dtype=weight_dtype)
    t5_tokens = t5_tokenizer(prompt, padding="max_length", max_length=512,
                             truncation=True, return_tensors="pt").input_ids.to(device)
    prompt_embeds = t5_encoder(t5_tokens)[0].to(dtype=weight_dtype)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)

    guidance = torch.full([1], guidance, device=device, dtype=torch.float32)
    guidance = guidance.expand(img.shape[0])
    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "prompt_embeds": prompt_embeds.to(img.device),
        "text_ids": txt_ids.to(img.device),
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "guidance": guidance,
    }

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)
    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()

def denoising_euler(
    transformer, latents, sigmas, guidance,
    pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
    for k in range(len(sigmas)-1):
        sigma_k, sigma_next = sigmas[k], sigmas[k+1]
        t_in = torch.full((latents.shape[0],), sigma_k, device=latents.device, dtype=latents.dtype)
        v = transformer(
            hidden_states=latents,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            return_dict=False,
        )[0]
        latents = latents + (sigma_next-sigma_k)*v
    return latents

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def vae_decode(latents, vae, image_processor,):
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    with torch.amp.autocast('cuda', enabled=False):
        vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32).contiguous()
        image_tensor = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image_tensor.detach().cpu(), output_type="pil")
    return image
