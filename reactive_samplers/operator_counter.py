from collections import Counter
from contextlib import contextmanager
from functools import wraps
import math
import sys
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import FlopCounterMode


def elementwise(*args, out_shape=None, **kwargs):
    return math.prod(out_shape)


def sdpa_forward(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs):
    # GQA shares K/V heads; QK and AV still run for every query head.
    batch = math.prod(torch.broadcast_shapes(query_shape[:-3], key_shape[:-3], value_shape[:-3]))
    heads, queries, width = query_shape[-3:]
    keys, value_width = key_shape[-2], value_shape[-1]
    return 2 * batch * heads * queries * keys * (width + value_width)


class ShapeRecorder(TorchDispatchMode):
    def __init__(self, registry):
        super().__init__()
        self.registry = registry
        self.shapes = Counter()
        self.unregistered = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        inputs = tuple((tuple(t.shape), str(t.dtype))
                       for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor))
        outputs = tuple((tuple(t.shape), str(t.dtype))
                        for t in tree_flatten(result)[0] if isinstance(t, torch.Tensor))
        self.shapes[(str(func), inputs, outputs)] += 1
        if func.overloadpacket not in self.registry:
            self.unregistered[str(func)] += 1
        return result


class OperatorCounter:
    def __init__(self):
        mapping = {getattr(torch.ops.aten, op): elementwise
                   for op in ("add", "sub", "mul", "div", "neg", "abs")}
        mapping.update({getattr(torch.ops.aten, op): sdpa_forward for op in (
            "scaled_dot_product_attention", "_scaled_dot_product_efficient_attention",
            "_scaled_dot_product_flash_attention", "_scaled_dot_product_cudnn_attention",
            "_scaled_dot_product_flash_attention_for_cpu")})
        self.counter = FlopCounterMode(display=False, custom_mapping=mapping)
        self.recorder = ShapeRecorder(self.counter.flop_registry)
        self.external_attention = 0
        self.attention_shapes = Counter()

    @contextmanager
    def wan_attention(self, model):
        module = sys.modules[type(model).__module__]
        original = module.flash_attention

        @wraps(original)
        def attention(q, k, v, *args, **kwargs):
            q_lens, k_lens = kwargs.get("q_lens"), kwargs.get("k_lens")
            nq = [q.shape[1]] * q.shape[0] if q_lens is None else q_lens.tolist()
            nk = [k.shape[1]] * k.shape[0] if k_lens is None else k_lens.tolist()
            if kwargs.get("causal", False) or kwargs.get("window_size", (-1, -1)) != (-1, -1):
                raise ValueError("The existing Wan attention must be dense and noncausal")
            # Count the actual QK and AV matrix shapes once per attention call.
            self.external_attention += 2 * sum(a * b for a, b in zip(nq, nk)) * q.shape[2] * (q.shape[3] + v.shape[3])
            self.attention_shapes[(tuple(q.shape), tuple(k.shape), tuple(v.shape),
                                   tuple(nq), tuple(nk))] += 1
            with _disable_current_modes():
                return original(q, k, v, *args, **kwargs)

        with patch.object(module, "flash_attention", attention):
            yield

    def run(self, function):
        with self.counter, self.recorder:
            result = function()
        return result

    def record(self):
        registered = self.counter.get_total_flops()
        return dict(
            tflops=(registered + self.external_attention) / 1e12,
            registered_flops=registered, external_attention_flops=self.external_attention,
            operator_shapes=[dict(op=op, inputs=inputs, outputs=outputs, calls=calls)
                             for (op, inputs, outputs), calls in self.recorder.shapes.items()],
            attention_shapes=[dict(q=q, k=k, v=v, q_lengths=nq, k_lengths=nk, calls=calls)
                              for (q, k, v, nq, nk), calls in self.attention_shapes.items()],
            unregistered_operators=dict(self.recorder.unregistered),
            multiply_add_flops=2,
            scope="Registered matrix/convolution/SDPA and add/sub/mul/div/neg/abs, plus external Wan attention QK/AV. Unregistered operations are excluded and listed.",
        )
