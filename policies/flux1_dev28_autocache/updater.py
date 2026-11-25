def choose(observation, signals, state):
    step = state.get("step", 0)
    state["step"] = step + 1
    if step == 9:
        variation = signals["feature_relative_cv"]
        state["early_tail"] = variation is not None and variation < 0.19
    if step == 19 and not state.get("early_tail", False):
        movement = observation["feature_abs_change"]
        state["late_tail"] = movement is not None and movement < 0.014
    if state.get("early_tail", False):
        nodes = (0, 1, 2, 3, 4, 6, 9, 13, 18, 25)
    elif state.get("late_tail", False):
        nodes = (0, 1, 2, 3, 4, 6, 9, 13, 20, 25)
    else:
        nodes = (0, 1, 2, 3, 4, 6, 9, 13, 19, 25)
    refresh = not observation["has_reference"] or step in nodes
    return SkipAction(not refresh)
