"""Qwen evaluation using the unchanged batch sampler and metric worker."""
import hashlib
import json
from pathlib import Path
import time

from reactive_samplers.report import write_json
from .policy import Runtime
from .selection import summarize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def validate_rows(rows):
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids)) != len(ids): raise ValueError('Samples must be nonempty with unique IDs')
    for r in rows:
        if not isinstance(r['id'], str) or not r['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in r['id']):
            raise ValueError('Sample IDs must be safe filename components')
        if type(r['seed']) is not int or not isinstance(r['prompt'], str): raise ValueError('Prompt/seed missing')


def valid_result(path, expected=None):
    data = read(path)
    if expected is not None and any(data['context'].get(k) != v for k, v in expected.items()
                                    if k not in ('policy_sha256', 'reference_sha256')):
        raise ValueError('Evaluation configuration changed')
    if len(data['images']) != len(data['context']['rows']): raise ValueError('Incomplete evaluation')
    for row, sample in zip(data['images'], data['context']['rows']):
        if any(row[k] != sample[k] for k in ('id', 'prompt', 'seed')): raise ValueError('Sample alignment changed')
    return data


class Evaluator:
    def __init__(self, generation, batch_size, state_bytes):
        self.config, self.batch_size, self.state_bytes = generation, batch_size, state_bytes
        self.device = generation['device']
        self.adapter = self.metrics = None

    def load(self):
        if self.adapter is not None: return
        if self.config['model'] in ('flux3', 'flux'):
            if self.config['model'] == 'flux3':
                from .flux3 import load_adapter
            else:
                from .flux1 import load_adapter
            from reactive_samplers.qwen_metrics import LegacyMetrics
            self.adapter = load_adapter(self.config, self.device)
            self.metrics = LegacyMetrics(self.config['metrics_python'], self.config['metrics'])
            return
        import torch
        from diffusers import QwenImage21Pipeline
        from reactive_samplers.qwen_batch import BatchAdapter
        from reactive_samplers.qwen_metrics import LegacyMetrics
        g = self.config
        pipe = QwenImage21Pipeline.from_pretrained(g['model_path'], torch_dtype=getattr(torch, g['dtype']), local_files_only=True).to(self.device)
        pipe.transformer.eval().requires_grad_(False)
        self.adapter = BatchAdapter(pipe, **{k:g[k] for k in ('guidance_scale','height','width','max_sequence_length')})
        self.metrics = LegacyMetrics(g['metrics_python'], g['metrics'])

    def context(self, rows, policy, initial, reference):
        return {'generation': self.config, 'batch_size': self.batch_size, 'state_bytes': self.state_bytes,
                'rows': rows, 'initial': initial}

    def run(self, rows, policy, output, reference=None, initial=False, probe=None):
        import torch
        if self.config['model'] == 'flux3':
            from .flux3 import sample_batch
        elif self.config['model'] == 'flux':
            from .flux1 import sample_batch
        else:
            from reactive_samplers.qwen_batch import sample_batch
        rows = list(rows); validate_rows(rows); output = Path(output)
        context = self.context(rows, policy, initial, reference)
        if (output/'results.json').exists() and probe is None:
            return valid_result(output/'results.json', context)
        output.mkdir(parents=True, exist_ok=True)
        if (output/'context.json').exists() and read(output/'context.json') != context:
            raise ValueError('Output directory belongs to another evaluation')
        write_json(output/'context.json', context)
        references = {}
        if reference:
            ref = valid_result(reference)
            if not ref['context']['initial'] or ref['context']['generation'] != self.config or ref['context']['batch_size'] != self.batch_size or ref['context']['rows'] != rows:
                raise ValueError('Reference must match the same all-NEW generation, batch and rows')
            if any(r['nfe'] != self.config['reference_steps'] * (2 if self.config['model'] == 'flux3' else 1) for r in ref['images']): raise ValueError('Reference is not all-NEW')
            references = {r['id']: r for r in ref['images']}
        elif not initial:
            raise ValueError('A fixed all-NEW reference is required')
        all_rows, batches = [], []
        for start in range(0, len(rows), self.batch_size):
            group = rows[start:start+self.batch_size]
            batch = output/'batches'/f'{start//self.batch_size+1:03d}'
            batch_context = {**context, 'rows': group, 'batch_start': start}
            if (batch/'results.json').exists() and probe is None:
                saved = valid_result(batch/'results.json', batch_context)
                all_rows.extend(saved['images']); batches.append(saved['timing']); continue
            batch.mkdir(parents=True, exist_ok=True)
            self.load()
            runtime = Runtime(policy, self.state_bytes, probe, self.config.get('observation_fields'))
            self.adapter.synchronize()
            device = torch.device(self.adapter.device)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            began = time.perf_counter()
            latent, traces, timing = sample_batch(self.adapter, group, runtime, self.config['reference_steps'],
                max_consecutive_skips=self.config['reference_steps'], timeout=self.config.get('policy_timeout_seconds',30))
            decoded = time.perf_counter()
            images = self.adapter.finish_batch(latent)
            self.adapter.synchronize()
            seconds = time.perf_counter()-began
            peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
            del latent, runtime
            records = []
            if len(images) != len(group) or len(traces) != len(group): raise ValueError('Incomplete batch output')
            for row, image, trace, nfe in zip(group, images, traces, timing['nfe']):
                if len(trace) != self.config['reference_steps']: raise ValueError('Incomplete trajectory')
                image_path = batch/(row['id']+'.png'); image.save(image_path)
                trajectory = batch/(row['id']+'.jsonl')
                trajectory.write_text(''.join(json.dumps(t,allow_nan=False)+'\n' for t in trace))
                reference_path = references[row['id']]['image'] if references else image_path
                scores = self.metrics.score(image_path, row['prompt'], reference_path)
                records.append({**row, 'image': str(image_path.resolve()), 'image_sha256': file_hash(image_path),
                    'trajectory': str(trajectory.resolve()), 'trajectory_sha256': file_hash(trajectory),
                    'observations_available': True,
                    'scores': scores, 'nfe': nfe, 'seconds': seconds/len(group),
                    'sampling_seconds': timing['batch_sampling_seconds']/len(group)})
            summary = summarize(records, initial)
            batch_timing = {**timing, 'generation_seconds': seconds,
                'decode_seconds': seconds-(decoded-began),
                'peak_allocated_bytes': peak, 'time_kind': 'amortized batch generation, including signal extraction; excludes scoring and Agent'}
            write_json(batch/'results.json', {'context': batch_context, 'images': records, 'summary': summary, 'timing': batch_timing})
            all_rows.extend(records); batches.append(batch_timing)
            write_json(output/'progress.json', {'completed_images': len(all_rows), 'total_images': len(rows)})
        data = {'context': context, 'images': all_rows, 'summary': summarize(all_rows, initial), 'batches': batches}
        write_json(output/'results.json', data)
        return data

    def close(self):
        if self.metrics is not None:
            self.metrics.close()
            self.metrics = None
        if self.adapter is not None:
            self.adapter = None
            import torch
            torch.cuda.empty_cache()
