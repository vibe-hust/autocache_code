"""Restricted tensor signals and scalar decisions over the unchanged sampler API."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace, MappingProxyType

from reactive_samplers.context import SkipAction, validate_skip
from reactive_samplers.report import write_json
from reactive_samplers.policy_observations import analyze, definitions
from reactive_samplers.policy_tracing import TraceTests

INTERFACE = 'opal-tensor-signals-v1'
FILES = ('signals.py', 'updater.py')
TENSORS = ('latent', 'feature', 'previous_latent', 'previous_feature')
BUILTINS = dict(abs=abs, min=min, max=max, sum=sum, len=len, range=range, sorted=sorted,
                float=float, int=int, bool=bool, round=round, all=all, any=any, enumerate=enumerate)
MATH = {'sqrt', 'log', 'log1p', 'exp', 'isfinite', 'floor', 'ceil', 'tanh'}
DICT_METHODS = {'get', 'update', 'clear', 'pop', 'setdefault', 'append', 'keys', 'values', 'items'}
TENSOR_METHODS = {'abs', 'mean', 'sum', 'square', 'sqrt', 'norm', 'std', 'var', 'amax', 'amin',
                  'max', 'min', 'clamp', 'clamp_min', 'clamp_max', 'reshape', 'flatten', 'float',
                  'double', 'detach', 'clone', 'item', 'numel', 'quantile', 'median', 'pow',
                  'transpose', 'permute', 'unsqueeze', 'squeeze', 'contiguous', 'size'}


def tree(source, kind):
    parsed = ast.parse(source)
    name, args = ('extract_signals', ['tensors', 'state']) if kind == 'signals' else ('choose', ['observation', 'signals', 'state'])
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef):
        raise ValueError('Only one policy function is allowed')
    fn = parsed.body[0]
    if (fn.name != name or [a.arg for a in fn.args.args] != args or fn.decorator_list or fn.returns
            or fn.args.defaults or fn.args.kwonlyargs or fn.args.posonlyargs or fn.args.vararg or fn.args.kwarg
            or any(a.annotation for a in fn.args.args)):
        raise ValueError('Policy function signature mismatch')
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                 ast.Lambda, ast.With, ast.Try, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)
    methods = DICT_METHODS | (TENSOR_METHODS if kind == 'signals' else set())
    for node in ast.walk(fn):
        if isinstance(node, forbidden) or isinstance(node, ast.FunctionDef) and node is not fn:
            raise ValueError('Unsupported policy behavior')
        if isinstance(node, ast.Name) and node.id.startswith('_'):
            raise ValueError('Private names are unavailable')
        if isinstance(node, ast.Attribute):
            math_call = isinstance(node.value, ast.Name) and node.value.id == 'math' and node.attr in MATH
            if not isinstance(node.ctx, ast.Load) or not (math_call or node.attr in methods):
                raise ValueError('Unsupported attribute: '+node.attr)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in {*BUILTINS, 'SkipAction'}:
                    raise ValueError('Unsupported call')
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError('Indirect calls are unavailable')
            if any(k.arg is None or k.arg in ('out', 'inplace') for k in node.keywords):
                raise ValueError('Output mutation is unavailable')
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in args:
                    raise ValueError('Cannot rebind interface arguments')
                if isinstance(node, ast.AugAssign) and not isinstance(target, ast.Name):
                    raise ValueError('In-place indexed operations are unavailable')
    return parsed


def source_hash(folder):
    h = hashlib.sha256()
    for name in FILES:
        path = Path(folder)/name
        if path.is_symlink(): raise ValueError('Policy symlinks are unavailable')
        content = path.read_bytes()
        if len(content) > 100_000: raise ValueError('Policy source is too large')
        tree(content.decode('utf-8'), name[:-3])
        h.update(name.encode()+b'\0'+content+b'\0')
    return h.hexdigest()


def structure_hash(folder):
    return hashlib.sha256(''.join(ast.dump(tree((Path(folder)/n).read_text(), n[:-3]), include_attributes=False) for n in FILES).encode()).hexdigest()


def freeze(source, target, model='qwen21'):
    source, target = Path(source), Path(target)
    if set(p.name for p in source.iterdir()) - {*FILES, 'manifest.json'}:
        raise ValueError('Only the two policy files and manifest may be submitted')
    version = source_hash(source)
    target.mkdir(parents=True, exist_ok=False)
    for name in FILES: (target/name).write_bytes((source/name).read_bytes())
    write_json(target/'manifest.json', {'interface': INTERFACE, 'model': model,
        'method': 'residual_cache', 'sha256': version, 'structure_sha256': structure_hash(source)})
    return target


def scalar_map(value):
    if type(value) is not dict or any(type(k) is not str or k.startswith('_') for k in value):
        raise ValueError('Signals must be a dictionary with public string keys')
    if len(value) > 128 or any(v is not None and (type(v) not in (int, float, bool) or not math.isfinite(v)) for v in value.values()):
        raise ValueError('Signals must be finite Python scalars or null')
    return value


class Runtime:
    """Duck-typed Programs adapter; shared sample_batch and A loader are untouched."""
    def __init__(self, folder, state_bytes=64*1024*1024, probe=None, observation_fields=None):
        folder = Path(folder)
        self.version = source_hash(folder)
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest.get('interface') != INTERFACE or manifest.get('sha256') != self.version:
            raise ValueError('Frozen policy manifest mismatch')
        self.limit, self.probe = state_bytes, probe
        self.observation_fields = observation_fields
        self.states, self.trace = {}, {}
        self.tests, self.operands, self.returned = [], {}, []
        self.extract = self.compile((folder/'signals.py').read_text(), 'signals')
        self.decide = self.compile((folder/'updater.py').read_text(), 'updater')
        self.analysis = SimpleNamespace(analyze=self.analyze)
        self.updater = SimpleNamespace(choose=self.choose)

    def compile(self, source, kind):
        parsed = tree(source, kind)
        namespace = {'__builtins__': BUILTINS, 'math': math, 'SkipAction': SkipAction,
                     '_operand': self.operand, '_test': self.test, '_guard': self.guard, '_returned': self.return_value}
        if kind == 'updater': parsed = TraceTests().visit(parsed)
        exec(compile(ast.fix_missing_locations(parsed), '<opal-'+kind+'>', 'exec'), namespace)
        return namespace['extract_signals' if kind == 'signals' else 'choose']

    def operand(self, expression, index, value):
        if index == 0: self.operands[expression] = []
        self.operands[expression].append(value)
        return value

    def test(self, expression, value):
        self.tests.append({'expression': expression, 'values': self.operands.pop(expression), 'result': bool(value)})
        return value

    def guard(self, expression, value):
        self.tests.append({'expression': 'if '+expression, 'values': [bool(value)], 'result': bool(value)})
        return value

    def return_value(self, expression, value):
        self.returned.append(expression)
        return value

    def analyze(self, ctx, unused_memory):
        values = analyze(ctx)
        return values if self.observation_fields is None else {k: values[k] for k in self.observation_fields}

    def check_state(self, state):
        import torch
        seen = set(); size = 0
        def visit(value, depth=0):
            nonlocal size
            if depth > 12: raise ValueError('State nesting limit exceeded')
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage(); key = (str(value.device), storage.data_ptr())
                if key not in seen: size += storage.nbytes(); seen.add(key)
            elif type(value) is dict:
                if len(value) > 256 or any(type(k) is not str for k in value): raise ValueError('Invalid state keys')
                for item in value.values(): visit(item, depth+1)
            elif type(value) in (list, tuple):
                if len(value) > 256: raise ValueError('State item limit exceeded')
                for item in value: visit(item, depth+1)
            elif value is not None and (type(value) not in (bool, int, float) or not math.isfinite(value)):
                raise ValueError('State supports finite scalars, tensors and containers only')
        visit(state)
        if size > self.limit: raise ValueError('Per-image retained tensor-state budget exceeded')
        return size

    def choose(self, observation, ctx, unused_memory):
        import torch
        self.tests.clear(); self.operands.clear(); self.returned.clear()
        if ctx.step == 0: self.states[ctx.sample_id] = {'scalars': {}, 'tensors': {}}
        state = self.states[ctx.sample_id]
        tensors = dict(zip(TENSORS, (ctx.latent, ctx.input_feature, ctx.reference_latent, ctx.reference_feature)))
        # Clones prevent policy operations, including aliases and +=, changing the sampler.
        tensors = {k: None if v is None else v.detach().clone() for k, v in tensors.items()}
        probe_result = self.probe(ctx, tensors) if self.probe else None
        signals = scalar_map(self.extract(MappingProxyType(tensors), state))
        self.check_state(state)
        # The decision function can update scalar state; tensor history stays in extraction.
        action = self.decide(MappingProxyType(observation), MappingProxyType(signals), state['scalars'])
        validate_skip(action)
        json.dumps(state['scalars'], allow_nan=False)
        state_bytes = self.check_state(state)
        self.trace = {'signals': copy.deepcopy(signals), 'tests': copy.deepcopy(self.tests),
            'return_expression': self.returned[-1], 'requested_action': 'reuse' if action.skip else 'new',
            'state_tensor_bytes': state_bytes}
        if probe_result is not None: self.trace['diagnostic'] = probe_result
        json.dumps(self.trace, allow_nan=False)
        return action


def interface_description(model='qwen21', observation_fields=None):
    ctx = SimpleNamespace(method='residual_cache', model=model, observation_reference_step=None, observation_reference_sigma=None)
    spec = definitions(ctx)
    if observation_fields is not None: spec['fields'] = {k: spec['fields'][k] for k in observation_fields}
    return {'interface': INTERFACE, 'observations': spec,
        'tensor_names': list(TENSORS), 'tensor_shape': 'Per-image tensors retain batch dimension 1; latent is the model latent, feature is post-input-embedding and pre-modulation. Previous tensors refer to the adjacent pre-update node; null initially.',
        'state': 'extract_signals receives {scalars: {}, tensors: {}}; choose receives its scalars dictionary. Reset per image. Retained tensor bytes are bounded by configuration.',
        'tensor_methods': sorted(TENSOR_METHODS), 'scalar_builtins': sorted(BUILTINS)}
