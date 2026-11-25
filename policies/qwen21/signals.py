def extract_signals(tensors, state):
    if state['scalars'].get('branch_chosen', False):
        state['tensors'].clear()
        return {'direction_change': None}
    previous = tensors['previous_latent']
    if previous is None:
        return {'direction_change': None}
    delta = tensors['latent'].float() - previous.float()
    magnitude = max(float(delta.square().mean().sqrt().item()), 1e-8)
    old = state['tensors'].get('previous_delta')
    change = None
    if old is not None:
        old_magnitude = max(float(old.square().mean().sqrt().item()), 1e-8)
        change = float((delta / magnitude - old / old_magnitude).square().mean().sqrt().item())
    state['tensors']['previous_delta'] = delta.detach().clone()
    return {'direction_change': change}
