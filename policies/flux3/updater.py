def choose(observation, signals, state):
    k = state.get('node', 0)
    state['node'] = k + 1
    if k < 14:
        return SkipAction(False)
    if k == 14:
        cosine = signals.get('latent_cosine')
        extend = cosine is not None and cosine < 0.96
        if extend:
            state['schedule'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 21, 25, 29, 34, 39, 43, 47, 49)
        else:
            state['schedule'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 21, 25, 29, 33, 37, 41, 44, 47, 49)
    fresh = not observation['has_reference'] or k in state['schedule']
    return SkipAction(not fresh)
