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

    def restore(self, record):
        if self.capture:
            return False
        saved = []
        for line in Path(record['trajectory']).read_text().splitlines():
            event = json.loads(line)
            signals = event.get('reason', {}).get('diagnostic')
            if event['step'] in self.steps and signals is not None:
                saved.append({'sample_id': record['id'], 'step': event['step'], 'signals': signals})
        if {node['step'] for node in saved} != self.steps:
            return False
        self.records.extend(saved)
        return True

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


def diagnose(request, workspace, output, records, rows, evaluator, reference, config):
    record = next((r for r in records if r['id'] == request['strategy_id'] and r['status'] in ('completed','early_stopped')),None)
    if record is None: raise ValueError('Diagnostic requires an evaluated strategy')
    parent = read(record['results']) if record['status']=='early_stopped' else valid_result(record['results'])
    ids = request.get('sample_ids',[]); steps = request.get('steps',[])
    if not ids or not set(ids) <= {r['id'] for r in parent['videos']}: raise ValueError('Choose known diagnostic sample IDs')
    if not steps or any(type(s) is not int or s<0 or s>=config['generation']['reference_steps'] for s in steps):
        raise ValueError('Choose valid diagnostic steps')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if request['kind']=='trajectory':
        selected=[]
        for video in parent['videos']:
            if video['id'] not in ids: continue
            for line in Path(video['trajectory']).read_text().splitlines():
                event=json.loads(line)
                if event['step'] in steps: selected.append({'sample_id':video['id'],**event})
        result={'kind':'trajectory','nodes':selected,'extra_backbone_items':0}
    elif request['kind']=='probe':
        from .parallel import ParallelEvaluator
        source=safe_child(workspace,request['probe_path']).read_text()
        capture=request.get('capture_tensors',[])
        if not isinstance(capture,list) or not set(capture)<=set(TENSORS): raise ValueError('Invalid capture names')
        selected=[row for row in rows if row['id'] in ids]
        initial=record['id']=='initial'
        replay_reference=None if initial else reference
        if isinstance(evaluator, ParallelEvaluator):
            job={'id':'probe_'+record['id'], 'policy':record['policy'], 'output':str(output/'replay'),
                 'initial':initial, 'reference_rows':rows,
                 'probe':{'source':source, 'steps':steps, 'output':str(output),
                          'capture':capture, 'byte_limit':config['diagnostic_tensor_bytes']}}
            _,measured,error=next(evaluator.run_jobs([job],selected,replay_reference))
            if error is not None: raise ValueError(error)
            records=measured['diagnostic_records']; saved_bytes=measured['saved_tensor_bytes']
        else:
            probe=Probe(source,ids,steps,output,capture,config['diagnostic_tensor_bytes'])
            measured=evaluator.run(selected,record['policy'],output/'replay',reference=replay_reference,
                                   initial=initial,probe=probe,reference_rows=rows)
            records=probe.records; saved_bytes=probe.bytes
        if saved_bytes>config['diagnostic_tensor_bytes']: raise ValueError('Diagnostic capture budget exceeded')
        originals={row['id']:row for row in parent['videos']}
        unchanged=all(a['video_sha256']==originals[a['id']]['video_sha256'] and
                      a['nfe']==originals[a['id']]['nfe'] for a in measured['videos'])
        for a in measured['videos']:
            b=originals[a['id']]
            def actions(path):
                events = [json.loads(line) for line in Path(path).read_text().splitlines()]
                return [(e.get('full_cond', not e['reuse_cache']),
                         e.get('full_uncond', not e['reuse_cache'])) for e in events]
            unchanged=unchanged and actions(a['trajectory'])==actions(b['trajectory'])
        result={'kind':'probe','records':records,'unchanged_videos_and_actions':unchanged,
                'extra_backbone_items':sum(r['nfe'] for r in measured['videos']),
                'replay_generation_seconds':sum(b['generation_seconds'] for b in measured['batches']),
                'saved_tensor_bytes':saved_bytes}
        write_json(output/'result.json',result)
        if not unchanged: raise ValueError('Diagnostic changed videos/actions; excluded from research evidence')
    else: raise ValueError('Unknown diagnostic kind')
    write_json(output/'result.json',result)
    return result
