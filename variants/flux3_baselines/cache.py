"""DrawBench T2I only. Each image and each of its two CFG branches has a separate history."""
from contextlib import contextmanager, nullcontext
from functools import wraps
from math import isqrt
from pathlib import Path
from runpy import run_path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import silu

_sea = run_path(str(Path(__file__).resolve().parents[2] / "baselines/SeaCache/Wan2.1/util_seacache.py"))
apply_sea_with_scheduler, rel_l1 = _sea["apply_sea_with_scheduler"], _sea["rel_l1"]


@contextmanager
def cache(model, scheduler, method, threshold, coefficients, steps=50):
    original = model.forward
    first = model.mode_blocks["image"][0]
    histories = [dict(total=0., residuals=[]), dict(total=0., residuals=[])]
    events, calibration = [], []

    @wraps(original)  # The modular denoiser inspects the signature to pass img_ids.
    def forward(**kw):
        step, branch = divmod(len(events), 2)
        state = histories[branch]
        x, t = kw["hidden_states"], kw["timestep"]
        if branch and events[-1]["sigma"] != t.item():
            raise RuntimeError("Expected two sequential CFG calls at the same timestep")
        h = model.x_embedders["image"](x)
        temb = model.timestep_embedder(model.time_proj(t.to(x.dtype) * 1000).to(x.dtype))[:, None]
        temb = temb.expand(-1, h.shape[1], -1)
        modulation = model.early_stream_modulations["image"](silu(temb))
        probe = None
        forced = step in (0, steps - 1)
        if method == "dicache":
            probe = first(h, modulation, model.pos_embed(kw["img_ids"]))
            signal = probe
            forced |= step <= int(.2 * steps)
        else:
            shift, scale, _ = modulation.chunk(3, dim=-1)
            signal = (1 + scale) * first.norm(h) + shift
            if method == "seacache" and not forced:
                side = isqrt(h.shape[1])  # All DrawBench images are square.
                signal = apply_sea_with_scheduler(
                    signal.reshape(1, side, side, -1), scheduler, step,
                    power_exp=2., dims=(-2, -3), norm_mode="mean",
                ).reshape_as(h)
        distance = rel_l1(signal, state["signal"]) if step else None
        state["signal"] = signal
        if not forced and method != "calibrate":
            state["total"] += float(np.polyval(coefficients, distance)) if method == "teacache" else distance
        reuse = method != "calibrate" and not forced and state["total"] < threshold
        residuals = state["residuals"]
        if reuse:
            delta = residuals[-1][0]
            if probe is not None and len(residuals) == 2:
                (old, p0), (recent, p1) = residuals
                gamma = ((probe-h-p0).abs().mean() /
                         (p1-p0).abs().mean()).clamp(1, 1.5)
                delta = old + gamma.to(old.dtype) * (recent-old)
            output = (model.proj_out(model.norm_out(h + delta, temb)),)
        else:
            def save_residual(module, args):
                residuals.append((args[0] - h, None if probe is None else probe - h))
                del residuals[:-2]
            # Continue a DiCache refresh from the already computed first block.
            use_probe = patch.object(first, "forward", lambda *a, **k: probe) if probe is not None else nullcontext()
            with use_probe, model.norm_out.register_forward_pre_hook(save_residual):
                output = original(**kw)
            if method == "calibrate":
                velocity = output[0]
                if step:
                    calibration.append([distance, rel_l1(velocity.float(), state["velocity"].float())])
                state["velocity"] = velocity
            state["total"] = 0.
        events.append(dict(step=step, cfg_branch=branch, sigma=t.item(), reuse=reuse))
        return output

    with patch.object(model, "forward", forward):
        yield events, calibration
    if len(events) != steps * 2:
        raise RuntimeError(f"Expected {steps * 2} CFG calls, observed {len(events)}")
