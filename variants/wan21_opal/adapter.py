"""Run-local Wan block wrappers, leaving the official CFG and scheduler in charge."""
import copy
from types import SimpleNamespace

from reactive_samplers.context import validate_skip
from reactive_samplers.recovery import policy_deadline, require_finite


class ResidualAdapter:
    def __init__(self, model, runtime, row, timesteps, sigmas, timeout=30):
        self.model, self.runtime, self.row = model, runtime, row
        self.timesteps = tuple(float(t) for t in timesteps)
        self.sigmas = tuple(float(s) for s in sigmas)
        if len(self.sigmas) != len(self.timesteps) + 1:
            raise ValueError('Expected one terminal sigma')
        self.timeout = timeout
        self.calls = self.nfe = 0
        self.age = 0
        self.branch_events = []
        self.residuals = [None, None]
        self.previous_latent = None
        self.previous_feature = None
        self.last_refresh_sigma = None
        self.trace = []

    def enter_model(self, module, args, kwargs):
        self.step, self.branch = divmod(self.calls, 2)
        if self.step >= len(self.timesteps):
            raise RuntimeError('Unexpected extra Wan model call')
        t = kwargs['t'] if 't' in kwargs else args[1]
        if abs(float(t.item()) - self.timesteps[self.step]) > 1e-4:
            raise RuntimeError('Wan timestep differs from the declared schedule')
        inputs = args[0] if args else kwargs['x']
        if len(inputs) != 1:
            raise ValueError('Wan video policy supports one video per sampler')
        self.current_latent = inputs[0].unsqueeze(0)

    def choose(self, feature):
        require_finite(feature, 'Wan input feature')
        sigma, next_sigma = self.sigmas[self.step:self.step + 2]
        last_sigma = self.last_refresh_sigma
        has_reference = all(residual is not None for residual in self.residuals)
        ctx = SimpleNamespace(
            sample_id=self.row['id'], step=self.step,
            model='wan21', method='residual_cache',
            latent=self.current_latent, input_feature=feature,
            reference_latent=self.previous_latent, reference_feature=self.previous_feature,
            cache_observation={'sigma': sigma, 'step_delta_sigma': sigma - next_sigma,
                'since_real_delta_sigma': 0. if last_sigma is None else last_sigma - sigma,
                'consecutive_skips': self.age, 'has_reference': has_reference})
        with policy_deadline(self.timeout):
            self.observation = self.runtime.analyze(ctx, {})
            requested = validate_skip(self.runtime.choose(self.observation, ctx, {}))
        self.reuse = requested and has_reference
        self.reason = {'requested_skip': requested, 'forced': None if has_reference else 'initialization',
                       'reason': copy.deepcopy(self.runtime.trace)}
        self.next_previous_latent = self.current_latent.detach().clone()
        self.next_previous_feature = feature.detach().clone()

    def wrap(self, index, original):
        def forward(x, *args, **kwargs):
            if index == 0:
                if self.branch == 0:
                    self.choose(x)
                if self.reuse:
                    return x + self.residuals[self.branch]
                self.backbone_input = x.detach().clone()
                self.nfe += 1
            if self.reuse:
                return x
            output = original(x, *args, **kwargs)
            if index == len(self.model.blocks) - 1:
                self.residuals[self.branch] = (output - self.backbone_input).detach()
                self.backbone_input = None
            return output
        return forward

    def leave_model(self, module, args, output):
        self.branch_events.append({'step': self.step, 'branch': self.branch,
            'sigma': self.sigmas[self.step], 'timestep': self.timesteps[self.step],
            'next_sigma': self.sigmas[self.step + 1], 'reuse_cache': self.reuse,
            'backbone_evaluated': not self.reuse, 'nfe': self.nfe,
            'version': self.runtime.version})
        if self.branch == 1:
            self.age = self.age + 1 if self.reuse else 0
            if not self.reuse:
                self.last_refresh_sigma = self.sigmas[self.step]
            self.previous_latent = self.next_previous_latent
            self.previous_feature = self.next_previous_feature
            cond, uncond = self.branch_events
            full_cond, full_uncond = not cond['reuse_cache'], not uncond['reuse_cache']
            self.trace.append({'step': self.step, 'sigma': self.sigmas[self.step],
                'timestep': self.timesteps[self.step], 'next_sigma': self.sigmas[self.step + 1],
                'full_cond': full_cond, 'full_uncond': full_uncond,
                'reuse_cache': not full_cond and not full_uncond,
                'step_nfe': int(full_cond) + int(full_uncond), 'nfe': self.nfe,
                'branches': self.branch_events, 'observation': self.observation,
                'version': self.runtime.version, **self.reason})
            self.branch_events = []
        self.calls += 1

    def __enter__(self):
        self.originals = [block.forward for block in self.model.blocks]
        if not self.originals:
            raise ValueError('Wan backbone has no blocks')
        for index, block in enumerate(self.model.blocks):
            block.forward = self.wrap(index, block.forward)
        self.handles = [self.model.register_forward_pre_hook(self.enter_model, with_kwargs=True),
                        self.model.register_forward_hook(self.leave_model)]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        for block, original in zip(self.model.blocks, self.originals):
            block.forward = original

    def validate_complete(self):
        if self.calls != 2 * len(self.timesteps) or len(self.trace) != len(self.timesteps):
            raise RuntimeError('Incomplete CFG trajectory')
