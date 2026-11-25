def extract_signals(tensors, state):
    result = {"feature_relative_cv": None}
    if state["scalars"].get("step", 0) == 9 and tensors["previous_feature"] is not None:
        previous = tensors["previous_feature"].float()
        difference = tensors["feature"].float() - previous
        token_absolute = difference.abs().mean(dim=-1).flatten()
        token_relative = token_absolute / (previous.abs().mean(dim=-1).flatten() + 1e-8)
        relative_mean = float(token_relative.mean().item())
        result["feature_relative_cv"] = float((token_relative - token_relative.mean()).square().mean().sqrt().item()) / max(relative_mean, 1e-8)
    return result
