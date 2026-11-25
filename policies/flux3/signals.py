def extract_signals(tensors, state):
    node = state['scalars'].get('node', 0)
    result = {'latent_cosine': None, 'latent_turn': None, 'feature_curvature_tail': None}
    if node != 13 and node != 14:
        return result
    history = state['tensors']
    for name in ['latent']:
        current = tensors[name]
        previous = tensors['previous_' + name]
        if previous is not None:
            if name == 'feature':
                current = current[:, ::4, ::4]
                previous = previous[:, ::4, ::4]
            delta = current.float() - previous.float()
            if node == 13:
                history[name + '_delta'] = delta.detach().clone()
            else:
                old_delta = history.get(name + '_delta')
                if old_delta is not None:
                    power = float(delta.square().mean().item())
                    old_power = float(old_delta.square().mean().item())
                    dot = float((delta * old_delta).mean().item())
                    if name == 'latent':
                        cosine = dot / max((power * old_power) ** 0.5, 1e-12)
                        result['latent_cosine'] = max(-1.0, min(1.0, cosine))
                        residual_power = max(0.0, power - dot * dot / max(old_power, 1e-12))
                        result['latent_turn'] = (residual_power / max(power, 1e-12)) ** 0.5
                    else:
                        projection = dot / max(old_power, 1e-12)
                        orthogonal = delta - old_delta * projection
                        local_curvature = orthogonal.square().mean(dim=-1).sqrt().flatten()
                        level = local_curvature.mean() * 0.0 + 0.9
                        local_tail = float(local_curvature.quantile(level).item())
                        result['feature_curvature_tail'] = local_tail / max(power ** 0.5, 1e-12)
    if node == 14:
        history.clear()
    return result
