def choose(observation, signals, state):
    node = state.get('node', 0)
    state['node'] = node + 1
    if not observation['has_reference'] or node < 6:
        return SkipAction(False)
    if not state.get('branch_chosen', False):
        direction = signals['direction_change']
        mode = 1
        change = observation['feature_rel_change']
        state['first_long_spacing'] = 5 if change is not None and change <= 0.027 else 6
        if direction is not None and direction <= 0.085:
            mode = 0
        elif direction is None or direction > 0.115 or change is None or change > 0.025:
            mode = 2
        state['mode'] = mode
        state['branch_chosen'] = True
    if state['mode'] == 2:
        if node < 9:
            return SkipAction(False)
        spacing = 8 if state.get('body_started', False) else state['first_long_spacing']
        tail = observation['sigma'] <= 0.16
        refresh = observation['consecutive_skips'] + 1 >= spacing
        if tail and not state.get('tail_refreshed', False):
            refresh = True
        if refresh:
            state['body_started'] = True
            if tail:
                state['tail_refreshed'] = True
        return SkipAction(not refresh)
    if state['mode'] == 0:
        if observation['sigma'] <= 0.16:
            refresh = not state.get('tail_refreshed', False)
            if refresh:
                state['tail_refreshed'] = True
            return SkipAction(not refresh)
        if state.get('cheap_body_calls', 0) >= 4:
            return SkipAction(True)
        spacing = state.get('spacing', 3)
        refresh = observation['consecutive_skips'] + 1 >= spacing
        if refresh:
            state['cheap_body_calls'] = state.get('cheap_body_calls', 0) + 1
            if spacing < 5:
                state['spacing'] = 5
            elif spacing < 6:
                state['spacing'] = 6
            else:
                state['spacing'] = 8
        return SkipAction(not refresh)
    if observation['sigma'] <= 0.20:
        refresh = observation['sigma'] <= 0.16 and not state.get('tail_refreshed', False)
        if refresh:
            state['tail_refreshed'] = True
        return SkipAction(not refresh)
    spacing = state.get('spacing', 3)
    refresh = observation['consecutive_skips'] + 1 >= spacing
    if refresh:
        state['spacing'] = min(spacing + 1, 6)
    return SkipAction(not refresh)
