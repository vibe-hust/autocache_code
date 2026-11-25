def choose(observation, signals, state):
    k = state.get('node', 0)
    state['node'] = k + 1
    reuse = observation['has_reference'] and k >= 6 and observation['consecutive_skips'] < 4
    return SkipAction(reuse)
