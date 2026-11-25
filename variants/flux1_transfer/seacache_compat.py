from typing import Any, Dict, Optional, Union
import argparse
import os
import re
from datetime import datetime
import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from variants.seacache.util_seacache import rel_l1, apply_sea_with_scheduler
logger = logging.get_logger(__name__)

def seacache_forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor=None, pooled_projections: torch.Tensor=None, timestep: torch.LongTensor=None, img_ids: torch.Tensor=None, txt_ids: torch.Tensor=None, guidance: torch.Tensor=None, joint_attention_kwargs: Optional[Dict[str, Any]]=None, controlnet_block_samples=None, controlnet_single_block_samples=None, return_dict: bool=True, controlnet_blocks_repeat: bool=False) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    Drop-in replacement for FluxTransformer2DModel.forward with SeaCache gating.
    Logic mirrors the denoise_cache gating (accumulated rescaled relative L1).
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop('scale', 1.0)
    else:
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif joint_attention_kwargs is not None and joint_attention_kwargs.get('scale', None) is not None:
        logger.warning('Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.')
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = self.time_text_embed(timestep, pooled_projections) if guidance is None else self.time_text_embed(timestep, guidance, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    image_rotary_emb = self.pos_embed(ids)
    should_calc = True
    if getattr(self, 'enable_seacache', False):
        inp = hidden_states
        temb_ = temb
        modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
        if self.cnt == 0 or self.cnt == self.num_steps - 1 or self.previous_modulated_input is None:
            should_calc = True
            self.accumulated_rel_l1_distance = 0.0
        else:
            modulated_inp = modulated_inp.reshape(modulated_inp.shape[0], int(img_ids[0, :, 1].max().item() + 1), int(img_ids[0, :, 2].max().item() + 1), modulated_inp.shape[-1])
            modulated_inp = apply_sea_with_scheduler(modulated_inp, self.scheduler, getattr(self, 'cnt', 0), power_exp=2.0, dims=(-2, -3), norm_mode='mean')
            modulated_inp = modulated_inp.reshape(modulated_inp.shape[0], -1, modulated_inp.shape[-1])
            self.accumulated_rel_l1_distance += rel_l1(modulated_inp, self.previous_modulated_input)
            if self.accumulated_rel_l1_distance < float(self.seacache_thresh):
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = modulated_inp
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    if getattr(self, 'enable_seacache', False) and (not should_calc) and (self.previous_residual is not None):
        hidden_states = hidden_states + self.previous_residual
    else:
        ori_hidden_states = hidden_states
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(block), hidden_states, encoder_hidden_states, temb, image_rotary_emb, **ckpt_kwargs)
            else:
                encoder_hidden_states, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
            if controlnet_block_samples is not None:
                interval_control = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb):
                        return module(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {'use_reentrant': False} if is_torch_version('>=', '1.11.0') else {}
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(block), hidden_states, encoder_hidden_states, temb, image_rotary_emb, **ckpt_kwargs)
            else:
                hidden_states = block(hidden_states=hidden_states, temb=temb, image_rotary_emb=image_rotary_emb)
            if controlnet_single_block_samples is not None:
                interval_control = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        if getattr(self, 'enable_seacache', False):
            self.previous_residual = hidden_states - ori_hidden_states
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
