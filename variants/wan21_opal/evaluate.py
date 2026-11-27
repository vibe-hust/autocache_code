"""Serial Wan sampling and paired frame metrics with per-video recovery."""
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time

from reactive_samplers.report import write_json
from .policy import Runtime, source_hash
from .selection import summarize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    # Direct reads avoid stalled CIFS page-cache reads during this run's recovery.
    if os.environ.get('OPAL_DIRECT_VIDEO_READ') == '1' and Path(path).suffix == '.mp4':
        command = ['dd', 'if=' + str(path), 'bs=1M', 'iflag=direct', 'status=none']
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as reader:
            for chunk in iter(lambda: reader.stdout.read(1024 * 1024), b''):
                h.update(chunk)
            error = reader.stderr.read()
            if reader.wait():
                raise OSError('Direct video read failed: ' + error.decode(errors='replace'))
    else:
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()


def validate_rows(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Nonempty unique sample IDs required')
    for row in rows:
        if not isinstance(row['id'], str) or not row['id'] or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in row['id']):
            raise ValueError('Unsafe sample ID')
        if type(row['seed']) is not int or not isinstance(row['prompt'], str):
            raise ValueError('Prompt/seed missing')


def validate_generation(g, check_paths=False):
    expected = {'model': 'wan21', 'method': 'residual_cache', 'dtype': 'bfloat16',
                'solver': 'unipc', 'batch_size': 1,
                'height': 480, 'width': 832, 'frames': 81, 'fps': 16,
                'shift': 8, 'guidance_scale': 6, 't5_cpu': False, 'offload_model': False}
    if any(g.get(k) != value for k, value in expected.items()):
        raise ValueError('Generation settings must match the existing Wan 1.3B references')
    if type(g.get('reference_steps')) is not int or g['reference_steps'] not in (40, 50):
        raise ValueError('Wan reference_steps must be 40 or 50')
    for key in ('model_path', 'official_source', 'dependency_site_packages') if check_paths else ():
        if not Path(g[key]).is_dir():
            raise ValueError('Missing ' + key)


def check_record(row, sample):
    if any(row[k] != sample[k] for k in ('id', 'prompt', 'seed')):
        raise ValueError('Sample alignment changed')


def valid_result(path, expected=None):
    data = read(path)
    if expected is not None and data['context'] != expected:
        raise ValueError('Evaluation provenance changed')
    if len(data['videos']) != len(data['context']['rows']):
        raise ValueError('Incomplete evaluation')
    for row, sample in zip(data['videos'], data['context']['rows']):
        check_record(row, sample)
    if summarize([dict(r) for r in data['videos']], initial=data['context']['initial']) != data['summary']:
        raise ValueError('Stored summary does not match its videos')
    return data


class Evaluator:
    def __init__(self, generation, batch_size, state_bytes):
        if batch_size != 1:
            raise ValueError('Wan search supports batch_size=1')
        self.config, self.batch_size, self.state_bytes = generation, batch_size, state_bytes
        self.pipe = self.metrics = None

    def context(self, rows, policy, initial, reference):
        return {'generation': self.config, 'batch_size': self.batch_size,
                'state_bytes': self.state_bytes, 'policy_sha256': source_hash(policy),
                'rows': rows, 'initial': initial,
                'reference_sha256': None if reference is None else file_hash(reference)}

    def load(self):
        if self.pipe is not None:
            return
        import torch
        validate_generation(self.config, check_paths=True)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one CUDA GPU to the video search')
        sys.path.insert(0, self.config['official_source'])
        if self.config['dependency_site_packages'] not in sys.path:
            sys.path.append(self.config['dependency_site_packages'])
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.modules import attention
        if not attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError('Reference comparison requires the same FA2 branch')
        self.pipe = wan.WanT2V(config=WAN_CONFIGS['t2v-1.3B'],
            checkpoint_dir=self.config['model_path'], t5_cpu=self.config['t5_cpu'])
        self.pipe.model.eval().requires_grad_(False)
        if self.pipe.param_dtype != torch.bfloat16:
            raise RuntimeError('Unexpected Wan precision')

    def prepare_generation(self, row, policy, probe=None):
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        from .adapter import ResidualAdapter

        g = self.config
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.pipe.num_train_timesteps,
                                              shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(g['reference_steps'], device=self.pipe.device, shift=g['shift'])
        runtime = Runtime(policy, self.state_bytes, probe)
        adapter = ResidualAdapter(self.pipe.model, runtime, row, scheduler.timesteps,
                                  scheduler.sigmas, g.get('policy_timeout_seconds', 30))

        def render():
            return self.pipe.generate(row['prompt'], size=(g['width'], g['height']),
                frame_num=g['frames'], shift=g['shift'], sample_solver=g['solver'],
                sampling_steps=g['reference_steps'], guide_scale=g['guidance_scale'],
                seed=row['seed'], offload_model=g['offload_model'])

        return adapter, render

    def generate(self, row, policy, folder, probe):
        import torch
        from wan.utils.utils import cache_video

        g = self.config
        adapter, render = self.prepare_generation(row, policy, probe)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode(), adapter:
            video = render()
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        adapter.validate_complete()
        if tuple(video.shape) != (3, g['frames'], g['height'], g['width']) or not torch.isfinite(video).all():
            raise RuntimeError('Invalid generated video')
        peak = torch.cuda.max_memory_allocated()
        path = folder / (row['id'] + '.mp4')
        pending = path.with_suffix('.pending.mp4')
        encoded = time.perf_counter()
        cache_video(video[None], save_file=str(pending), fps=g['fps'], nrow=1,
                    normalize=True, value_range=(-1, 1))
        encode_seconds = time.perf_counter() - encoded
        if not pending.is_file() or not pending.stat().st_size:
            raise RuntimeError('Video encoding failed')
        pending.replace(path)
        trajectory = folder / (row['id'] + '.jsonl')
        trajectory.write_text(''.join(json.dumps(t, allow_nan=False) + '\n' for t in adapter.trace))
        record = {**row, 'video': str(path.resolve()), 'video_sha256': file_hash(path),
            'trajectory': str(trajectory.resolve()), 'trajectory_sha256': file_hash(trajectory),
            'observations_available': True, 'nfe': adapter.nfe, 'seconds': seconds,
            'full_prediction_nodes': sum(t['full_cond'] and t['full_uncond'] for t in adapter.trace),
            'partial_prediction_nodes': sum(t['full_cond'] != t['full_uncond'] for t in adapter.trace), 'encode_seconds': encode_seconds,
            'peak_allocated_bytes': peak}
        del video, adapter
        return record

    def score(self, reference, candidate):
        from reactive_samplers.video.metrics import score
        if self.metrics is None:
            import lpips
            self.metrics = lpips.LPIPS(net='alex', verbose=False).eval().to('cpu')
        result = score(reference, candidate, self.metrics, 'cpu')
        if result['frames'] != self.config['frames']:
            raise ValueError('Scored video frame count mismatch')
        return {k: result[k] for k in ('psnr', 'ssim', 'lpips')}

    def _score_and_save(self, record, reference, result_path, context):
        record['scores'] = self.score(reference, record['video'])
        write_json(result_path, {'context_sha256': fingerprint(context), 'record': record})

    def run(self, rows, policy, output, reference=None, initial=False, probe=None, shared_context=None, reference_rows=None):
        rows, output = list(rows), Path(output)
        validate_rows(rows)
        context = shared_context if shared_context is not None else self.context(rows, policy, initial, reference)
        if shared_context is None and (output / 'results.json').exists() and probe is None:
            return valid_result(output / 'results.json', context)
        output.mkdir(parents=True, exist_ok=True)
        if shared_context is None:
            if (output / 'context.json').exists() and read(output / 'context.json') != context:
                raise ValueError('Output belongs to another evaluation')
            write_json(output / 'context.json', context)
        references = {}
        if reference:
            ref = valid_result(reference, self.context(reference_rows if reference_rows is not None else context['rows'],
                policy=Path(__file__).parent/'templates', initial=True, reference=None))
            if any(r['nfe'] != 2 * self.config['reference_steps'] for r in ref['videos']):
                raise ValueError('Reference must be all NEW')
            references = {r['id']: r for r in ref['videos']}
        elif not initial:
            raise ValueError('A fixed all-NEW reference is required')
        records, batches, pending_scores = [], [], []
        with ThreadPoolExecutor(max_workers=1) as scoring:
            for row in rows:
                folder = output / 'samples' / row['id']
                folder.mkdir(parents=True, exist_ok=True)
                result_path = folder / 'results.json'
                checkpoint = folder / 'generated.json'
                if result_path.exists() and (probe is None or probe.restore(read(result_path)['record'])):
                    saved = read(result_path)
                    record = saved['record']
                    if saved['context_sha256'] != fingerprint(context):
                        raise ValueError('Sample result context mismatch')
                    check_record(record, row)
                else:
                    if checkpoint.exists() and (probe is None or probe.restore(read(checkpoint)['record'])):
                        saved = read(checkpoint)
                        if saved['context_sha256'] != fingerprint(context):
                            raise ValueError('Generation checkpoint context mismatch')
                        record = saved['record']
                        check_record(record, row)
                    else:
                        self.load()
                        print('Generating', row['id'], flush=True)
                        record = self.generate(row, policy, folder, probe)
                        write_json(checkpoint, {'context_sha256': fingerprint(context), 'record': record})
                    if initial:
                        record['scores'] = {'psnr': None, 'ssim': 1., 'lpips': 0.}
                        write_json(result_path, {'context_sha256': fingerprint(context), 'record': record})
                    else:
                        pending_scores.append(scoring.submit(self._score_and_save, record,
                            references[row['id']]['video'], result_path, context))
                records.append(record)
                batches.append({'sample_ids': [row['id']], 'generation_seconds': record['seconds'],
                    'time_kind': 'Official generation including signal extraction and VAE decode; excludes scoring, encoding and Agent'})
                incomplete = 0
                for future in pending_scores:
                    if future.done():
                        future.result()
                    else:
                        incomplete += 1
                if shared_context is None:
                    write_json(output / 'progress.json', {'completed_videos': len(records) - incomplete,
                                                         'total_videos': len(rows)})
            for future in pending_scores:
                future.result()
        if shared_context is None:
            write_json(output / 'progress.json', {'completed_videos': len(records), 'total_videos': len(rows)})
        data = {'context': context, 'videos': records, 'summary': summarize(records, initial), 'batches': batches}
        if shared_context is None:
            write_json(output / 'results.json', data)
        return data

    def close(self):
        self.pipe = self.metrics = None
        gc.collect()
        if 'torch' in sys.modules:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
