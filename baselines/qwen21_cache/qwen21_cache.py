"""Independent Qwen-Image-2.1 baseline adaptations; official Pipeline stays intact."""
import math
from functools import partial

import torch


def relative_l1(current, previous):
    return float((current.float() - previous.float()).abs().mean() /
                 previous.float().abs().mean().clamp_min(1e-8))


class Qwen21Cache:
    """One image per context; never share this Pipeline across concurrent requests."""
    def __init__(self, transformer, method, steps, threshold, warmup_ratio=.2):
        if method not in ('teacache', 'dicache'):
            raise ValueError('method must be teacache or dicache')
        if steps < 2 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError('steps >= 2 and a finite nonnegative threshold are required')
        if not 0 <= warmup_ratio < 1:
            raise ValueError('warmup_ratio must be in [0, 1)')
        self.model, self.method, self.steps = transformer, method, steps
        self.threshold, self.warmup_ratio = threshold, warmup_ratio
        self.events, self.residuals, self.probe_residuals = [], [], []
        self.accumulated = 0.
        self.previous_signal = self.previous_probe = None
        self.nfe = self.block_evaluations = 0

    def __enter__(self):
        self.originals = [(block, 'forward' in block.__dict__, block.forward)
                          for block in self.model.transformer_blocks]
        for index, (block, _, original) in enumerate(self.originals):
            block.forward = partial(self._block, index, original)
        return self

    def __exit__(self, *exc):
        for block, overridden, original in self.originals:
            if overridden:
                block.forward = original
            else:
                del block.forward

    def _block(self, index, original, *args, **kwargs):
        # The official Qwen21 forward calls blocks with keyword arguments.
        hidden = kwargs['hidden_states']
        if index == 0:
            step = len(self.events)
            if step >= self.steps:
                raise RuntimeError('Use a new cache context for each image')
            if step == 0 and kwargs.get('kv_cache_mode') != 'extract':
                raise ValueError('Qwen21 baseline requires official Prefix KV prefill')
            self.target_mask = kwargs['target_token_mask']
            target = hidden[:, self.target_mask]
            self.input = target.detach()
            forced = step == 0 or step == self.steps - 1
            self.reuse = False
            signal_delta = None
            if self.method == 'teacache':
                block = self.model.transformer_blocks[0]
                mod1 = kwargs['modulation'].chunk(2, dim=-1)[0]
                signal, _ = block._modulate(block.img_norm1(hidden), mod1, self.target_mask)
                signal = signal[:, self.target_mask].detach()
                if not forced:
                    signal_delta = relative_l1(signal, self.previous_signal)
                    self.accumulated += signal_delta
                    self.reuse = self.accumulated < self.threshold
                self.previous_signal = signal
                output = None if self.reuse else original(*args, **kwargs)
            else:
                # Probe once; if refreshing, continue from this exact output.
                output = original(*args, **kwargs)
                probe = output[:, self.target_mask].detach()
                forced |= step <= int(self.warmup_ratio * self.steps)
                if not forced:
                    signal_delta = relative_l1(probe, self.previous_probe)
                    self.accumulated += signal_delta
                    self.reuse = self.accumulated < self.threshold
                self.previous_probe = probe
                self.probe_delta = probe - self.input
            if not self.reuse:
                self.accumulated = 0.
                self.nfe += 1
            self.events.append({'event': 'step', 'step': step, 'nfe': self.nfe,
                                'reuse_cache': self.reuse, 'backbone_evaluated': not self.reuse,
                                'signal_delta': signal_delta, 'accumulated_distance': self.accumulated,
                                'probe_blocks': int(self.method == 'dicache')})
            if self.reuse:
                delta = self.residuals[-1]
                if self.method == 'dicache' and len(self.residuals) == 2:
                    denominator = (self.probe_residuals[-1] - self.probe_residuals[-2]).float().abs().mean().clamp_min(1e-8)
                    gamma = ((self.probe_delta - self.probe_residuals[-2]).float().abs().mean() / denominator).clamp(1, 1.5)
                    delta = self.residuals[-2] + gamma.to(delta.dtype) * (self.residuals[-1] - self.residuals[-2])
                hidden = self.input + delta
                if self.method == 'dicache':
                    self.block_evaluations += 1
                return hidden
        elif self.reuse:
            return hidden
        else:
            output = original(*args, **kwargs)
        self.block_evaluations += 1
        if index == len(self.originals) - 1:
            self.residuals.append((output[:, self.target_mask] - self.input).detach())
            self.residuals = self.residuals[-2:]
            if self.method == 'dicache':
                self.probe_residuals.append(self.probe_delta)
                self.probe_residuals = self.probe_residuals[-2:]
        return output
