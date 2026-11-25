def choose(observation, signals, state):
    step = state.get('step', 0)
    state['step'] = step + 1
    if step == 2:
        state['early_feature'] = observation['feature_rel_change']
    if step == 10:
        early = state.get('early_feature')
        latent = observation['latent_rel_change']
        feature = observation['feature_rel_change']
        mode = 0
        if early is not None and latent is not None and feature is not None and early > 0 and latent > 0:
            if feature < 1.04 * latent:
                mode = -2
            elif feature < 1.31 * early or feature < 1.06 * latent:
                mode = -1
        state['mode'] = mode
    mode = state.get('mode', 0)
    if step == 23 and mode == 0:
        state['middle_latent'] = observation['latent_rel_change']
    if step == 27:
        latent = observation['latent_rel_change']
        base_tail = False
        if mode == -1:
            base_tail = latent is not None and latent >= 0.014
        elif mode == 0:
            middle = state.get('middle_latent')
            if middle is not None and latent is not None and middle > 0 and latent > 0:
                base_tail = latent < 1.27 * middle
        state['base_tail'] = base_tail
    warmup = 11 if mode == -2 else 10 if mode == -1 else 12
    period = 2 if step < 18 else 5 if step < 23 else 6 if step < 41 else 5
    if mode == -2:
        period = 2 if step < 16 else 5
    elif state.get('base_tail', False):
        period = 2 if step < 18 else 5 if step < 28 else 6
    refresh = (not observation['has_reference'] or step < warmup
               or step >= 48 or observation['consecutive_skips'] >= period - 1)
    return SkipAction(not refresh)
