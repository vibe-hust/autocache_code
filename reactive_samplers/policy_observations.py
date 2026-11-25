"""Framework-owned scalar observations. No model calls and no editable analysis."""
from pathlib import Path
import time

from reactive_samplers.observations import definitions as legacy_definitions

FIELDS = ('sigma', 'step_delta_sigma', 'since_real_delta_sigma', 'consecutive_skips',
          'has_reference', 'latent_abs_change', 'latent_rel_change',
          'feature_abs_change', 'feature_rel_change')
INTERFACE = 'fixed-observation-v1'
EPSILON = 1e-8


def changes(current, reference):
    if reference is None:
        return None, None
    current, reference = current.detach().float(), reference.detach().float()
    if current.shape != reference.shape:
        raise ValueError('Compared tensors must have the same shape')
    absolute = float((current-reference).abs().mean())
    scale = float(reference.abs().mean())
    return absolute, absolute / scale if scale > EPSILON else None


def analyze(ctx):
    latent_abs, latent_rel = changes(ctx.latent, ctx.reference_latent)
    feature_abs, feature_rel = changes(ctx.input_feature, ctx.reference_feature)
    return {**{key: ctx.cache_observation[key] for key in FIELDS[:5]},
            'latent_abs_change': latent_abs, 'latent_rel_change': latent_rel,
            'feature_abs_change': feature_abs, 'feature_rel_change': feature_rel}


def definitions(ctx):
    from types import SimpleNamespace
    result = legacy_definitions(ctx, SimpleNamespace(OBSERVATION_SPEC={}), dict.fromkeys(FIELDS[:5]))
    for tensor, symbol in [('latent', 'x'), ('feature', 'F')]:
        for kind in ('abs', 'rel'):
            result['fields'][f'{tensor}_{kind}_change'] = {
                'description': f'{"Mean absolute" if kind == "abs" else "Relative mean absolute"} change of {tensor} from the previous adjacent grid INPUT. Features are after input embedding, before timestep modulation. Measures distance, not signed growth or final image quality.',
                'mathematical_definition': f'mean(abs({symbol}[k]-{symbol}[k-1]))' + (f' / mean(abs({symbol}[k-1]))' if kind == 'rel' else ''),
                'reference': 'previous_adjacent_grid_input',
                'range_and_units': 'Nonnegative; '+('tensor units.' if kind == 'abs' else 'dimensionless ratio, not a percentage or a derivative per sigma.'),
                'missing_value': 'null when reference is absent'+('; also when reference mean absolute magnitude <= 1e-8.' if kind == 'rel' else '.'),
            }
    return result


DIAGNOSTIC_SPEC = {
    'reference_latent_abs_diff': {
        'description': 'Distance to the official all-refresh latent at the SAME pre-update node, prompt and seed. Agent evidence only; never a policy input.',
        'mathematical_definition': 'mean(abs(x[k]-x_official[k]))',
        'reference': 'matched_official_current_input', 'range_and_units': 'Nonnegative latent units.',
        'missing_value': 'null when reference diagnostics are disabled.'},
    'reference_latent_rel_diff': {
        'description': 'The same distance normalized by the official latent magnitude; not a change of the distance across time. Agent evidence only.',
        'mathematical_definition': 'mean(abs(x[k]-x_official[k])) / mean(abs(x_official[k]))',
        'reference': 'matched_official_current_input', 'range_and_units': 'Nonnegative dimensionless ratio.',
        'missing_value': 'null when disabled or reference mean absolute magnitude <= 1e-8.'},
}


class Observer:
    def __init__(self):
        self.configure({}, None)

    def configure(self, config, row):
        self.config, self.row = config, row
        self.current, self.last_step = {}, None
        self.seconds = 0.0

    def __call__(self, ctx, memory):
        observation = analyze(ctx)
        if ctx.step != self.last_step:
            began = time.perf_counter()
            diagnostic = {}
            if self.config.get('reference_latents'):
                import torch
                path = Path(self.config['reference_latents']) / self.row['id'] / f'{ctx.step:04d}.pt'
                saved = torch.load(path, map_location='cpu', weights_only=True)
                expected = {'prompt': ctx.prompt, 'seed': ctx.seed, 'step': ctx.step,
                            'sigma': ctx.sigma, 'sigmas': list(ctx.reference_sigmas)}
                if any(saved[key] != value for key, value in expected.items()):
                    raise ValueError('Reference latent node/prompt/seed/schedule mismatch')
                absolute, relative = changes(ctx.latent.cpu(), saved['latent'])
                diagnostic = dict(zip(DIAGNOSTIC_SPEC, (absolute, relative)))
            if self.config.get('record_reference_latents'):
                import torch
                path = Path(self.config['output_dir']) / 'reference_latents' / self.row['id'] / f'{ctx.step:04d}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({'prompt': ctx.prompt, 'seed': ctx.seed, 'step': ctx.step,
                            'sigma': ctx.sigma, 'sigmas': list(ctx.reference_sigmas),
                            'latent': ctx.latent.detach().cpu()}, path)
            self.current = {'diagnostic': diagnostic}
            self.last_step = ctx.step
            if self.config.get('reference_latents') or self.config.get('record_reference_latents'):
                self.seconds += time.perf_counter()-began
        return observation
