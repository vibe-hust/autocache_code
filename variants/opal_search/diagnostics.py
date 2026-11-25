"""On-demand trajectory summaries or non-intervening tensor probes."""
import ast
import json
import math
from pathlib import Path

from reactive_samplers.report import write_json
from .agent import safe_child
from .evaluate import read, valid_result
from .policy import tree, BUILTINS, scalar_map, TENSORS


class Probe:
    def __init__(self, source, samples, steps, output, capture, byte_limit):
        namespace = {'__builtins__': BUILTINS, 'math': math}
        exec(compile(ast.fix_missing_locations(tree(source,'signals')), '<diagnostic>', 'exec'), namespace)
        self.function = namespace['extract_signals']
        self.samples, self.steps, self.output, self.capture = set(samples), set(steps), Path(output), capture
        self.limit, self.bytes, self.states, self.records = byte_limit, 0, {}, []

    def __call__(self, ctx, tensors):
        if ctx.sample_id not in self.samples or ctx.step not in self.steps: return None
        import torch
        inputs = {k:None if v is None else v.clone() for k,v in tensors.items()}
        # Diagnostic state is separate from the strategy, with the same bounded schema.
        state = self.states.setdefault(ctx.sample_id, {'scalars':{},'tensors':{}})
        result = scalar_map(self.function(inputs, state))
        from .policy import Runtime
        Runtime.check_state(type('Budget',(),{'limit':self.limit})(), state)
        entry = {'sample_id':ctx.sample_id,'step':ctx.step,'signals':result}
        if self.capture:
            payload = {k:inputs[k] for k in self.capture}
            size = sum(v.numel()*v.element_size() for v in payload.values() if v is not None)
            if self.bytes+size > self.limit: raise ValueError('Diagnostic capture budget exceeded')
            self.bytes += size
            folder = self.output/'tensors'; folder.mkdir(parents=True,exist_ok=True)
            path = folder/f'{ctx.sample_id}-{ctx.step:03d}.pt'
            torch.save({k:None if v is None else v.cpu() for k,v in payload.items()},path)
            entry['capture'] = str(path)
        self.records.append(entry)
        return result


def validate_request(request, rows, config):
    if request['kind'] not in ('trajectory', 'probe'): raise ValueError('Unknown diagnostic kind')
    ids = request.get('sample_ids',[]); steps = request.get('steps',[])
    if not ids or not set(ids) <= {r['id'] for r in rows}: raise ValueError('Choose known diagnostic sample IDs')
    if not steps or any(type(s) is not int or s<0 or s>=config['generation']['reference_steps'] for s in steps):
        raise ValueError('Choose valid diagnostic steps')
    if request['kind']=='probe':
        capture=request.get('capture_tensors',[])
        if not isinstance(capture,list) or not set(capture)<=set(TENSORS): raise ValueError('Invalid capture names')
    return ids, steps


def diagnose(request, workspace, output, records, rows, evaluator, reference, config):
    record = next((r for r in records if r['id'] == request['strategy_id'] and r['status']=='completed'),None)
    if record is None: raise ValueError('Diagnostic requires a completed strategy')
    parent = valid_result(record['results'])
    ids, steps = validate_request(request, rows, config)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if request['kind']=='trajectory':
        selected=[]
        for image in parent['images']:
            if image['id'] not in ids: continue
            for line in Path(image['trajectory']).read_text().splitlines():
                event=json.loads(line)
                if event['step'] in steps: selected.append({'sample_id':image['id'],**event})
        result={'kind':'trajectory','nodes':selected,'extra_backbone_items':0}
    elif request['kind']=='probe':
        source=safe_child(workspace,request['probe_path']).read_text()
        capture=request.get('capture_tensors',[])
        probe=Probe(source,ids,steps,output,capture,config['diagnostic_tensor_bytes'])
        measured=evaluator.run(rows,record['policy'],output/'replay',reference=None if record['id']=='initial' else reference,
                               initial=record['id']=='initial',probe=probe)
        # Preserve the full batch composition: probe only selected nodes, replay the panel.
        unchanged=all(a['image_sha256']==b['image_sha256'] and a['nfe']==b['nfe'] for a,b in zip(measured['images'],parent['images']))
        for a,b in zip(measured['images'],parent['images']):
            actions=lambda p:[json.loads(line)['reuse_cache'] for line in Path(p).read_text().splitlines()]
            unchanged=unchanged and actions(a['trajectory'])==actions(b['trajectory'])
        result={'kind':'probe','records':probe.records,'unchanged_images_and_actions':unchanged,
                'extra_backbone_items':sum(r['nfe'] for r in measured['images']),
                'replay_generation_seconds':sum(b['generation_seconds'] for b in measured['batches']),
                'saved_tensor_bytes':probe.bytes}
        write_json(output/'result.json',result)
        if not unchanged: raise ValueError('Diagnostic changed images/actions; excluded from research evidence')
    else: raise ValueError('Unknown diagnostic kind')
    write_json(output/'result.json',result)
    return result
