"""Evaluate each strategy across available GPUs, serial samples on each GPU."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import os
import subprocess
from pathlib import Path
import sys

from reactive_samplers.report import write_json
from .evaluate import Evaluator, read, validate_rows, valid_result
from .selection import summarize


def gpu1_available(own_pids=()):
    try:
        result = subprocess.run(['nvidia-smi', '-i', '1', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        if pids - set(own_pids):
            return False
        if pids:
            return True
        result = subprocess.run(['nvidia-smi', '-i', '1', '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        memory, utilization = (int(value.strip()) for value in result.stdout.strip().split(','))
        return memory < 1024 and utilization <= 5
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _initialize(gpus, generation, batch_size, state_bytes, worker_dir):
    global _evaluator, _gpu, _status
    _gpu = gpus.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu)
    folder = Path(worker_dir)
    folder.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = (folder/f'gpu{_gpu}.log').open('a', buffering=1)
    _status = folder/f'gpu{_gpu}.json'
    if os.environ.get('WAN_TRUST_SAVED_RESULTS') == '1':
        from . import evaluate
        evaluate.file_hash = lambda path: None
        evaluate.valid_result = lambda path, expected=None: evaluate.read(path)
        evaluate.check_record = lambda row, sample: None
        evaluate.validate_generation = lambda generation: None
    _evaluator = Evaluator(generation, batch_size, state_bytes)
    write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'ready'})


def _evaluate(job, rows, reference, shared_context):
    write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'evaluation', 'strategy':job['id']})
    try:
        probe = None
        if 'probe' in job:
            from .diagnostics import Probe
            spec = job['probe']
            probe = Probe(spec['source'], [r['id'] for r in rows], spec['steps'],
                          spec['output'], spec['capture'], spec['byte_limit'])
        data = _evaluator.run(rows, job['policy'], job['output'], reference=reference,
                              initial=job.get('initial', False), probe=probe,
                              shared_context=shared_context, reference_rows=job.get('reference_rows'))
        if probe is not None:
            data.update(diagnostic_records=probe.records, saved_tensor_bytes=probe.bytes)
        write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'completed', 'strategy':job['id']})
        return data, None, os.getpid()
    except Exception as exc:
        error = f'{type(exc).__name__}: {str(exc)[-2000:]}'
        write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'failed', 'strategy':job['id'], 'error':error})
        return None, error, os.getpid()


class ParallelEvaluator(Evaluator):
    def __init__(self, generation, batch_size, state_bytes, gpus, worker_dir):
        super().__init__(generation, batch_size, state_bytes)
        self.gpus, self.worker_dir = gpus, worker_dir
        self.pool = self.gpu_queue = None
        self.active_gpus, self.worker_pids = [], set()

    def close_workers(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
            if self.gpu_queue is not None:
                self.gpu_queue.close()
            self.gpu_queue = None
            self.active_gpus, self.worker_pids = [], set()

    def run(self, *args, **kwargs):
        # Reference generation keeps its serial path; probes use run_jobs.
        self.close_workers()
        if os.environ.get('WAN_EVALUATION_BACKEND') == 'slurm':
            # The controller generates serial references on its first allocated GPU.
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
        return super().run(*args, **kwargs)

    def run_jobs(self, jobs, rows, reference):
        if not jobs:
            return
        rows = list(rows)
        validate_rows(rows)
        order = {row['id']: i for i, row in enumerate(rows)}
        for job in jobs:
            output = Path(job['output'])
            initial = job.get('initial', False)
            try:
                context = job.get('context') or self.context(rows, job['policy'], initial, reference)
                if (output/'results.json').exists():
                    yield job, valid_result(output/'results.json', context), None
                    continue
                output.mkdir(parents=True, exist_ok=True)
                if (output/'context.json').exists() and read(output/'context.json') != context:
                    raise ValueError('Output belongs to another evaluation')
                write_json(output/'context.json', context)
            except Exception as exc:
                yield job, None, f'{type(exc).__name__}: {str(exc)[-2000:]}'
                continue
            slurm = os.environ.get('WAN_EVALUATION_BACKEND') == 'slurm'
            selected = list(self.gpus) if slurm else [0, 1] if 1 in self.gpus and gpu1_available(self.worker_pids) else [0]
            if self.pool is not None and selected != self.active_gpus:
                self.close_workers()
            write_json(Path(self.worker_dir)/'dispatch.json', {'gpus':selected,
                'strategy':job['id'], 'sample_ids_by_gpu':{
                    str(gpu):[row['id'] for row in rows[i::len(selected)]] for i,gpu in enumerate(selected)},
                'rule':'Split one strategy by sample across Slurm-allocated GPUs' if slurm else
                       'Split one strategy by sample; GPU1 only when free of other compute processes'})
            print('Evaluating', job['id'], 'on GPUs', selected, flush=True)
            if self.pool is None:
                super().close()
                self.active_gpus = selected
                if slurm:
                    self.pool = ThreadPoolExecutor(max_workers=len(selected))
                else:
                    process_context = multiprocessing.get_context('spawn')
                    self.gpu_queue = process_context.Queue()
                    for gpu in selected: self.gpu_queue.put(gpu)
                    self.pool = ProcessPoolExecutor(max_workers=len(selected), mp_context=process_context,
                        initializer=_initialize, initargs=(self.gpu_queue, self.config, self.batch_size,
                            self.state_bytes, self.worker_dir))
            worker = _evaluate
            if slurm:
                from .slurm_parallel import evaluate
                worker = partial(evaluate, self.config, self.batch_size, self.state_bytes, self.worker_dir)
            futures = [self.pool.submit(worker, job, rows[i::len(selected)], reference, context)
                       for i in range(min(len(selected),len(rows)))]
            parts, errors = [], []
            for future in as_completed(futures):
                data, error, pid = future.result()
                self.worker_pids.add(pid)
                if error is not None:
                    errors.append(error)
                else:
                    parts.append(data)
                    write_json(output/'progress.json', {
                        'completed_videos':job.get('completed_count', 0) + sum(len(part['videos']) for part in parts),
                        'total_videos':len(context['rows'])})
            if errors:
                yield job, None, '; '.join(errors)
                continue
            videos = sorted((row for part in parts for row in part['videos']), key=lambda r:order[r['id']])
            batches = sorted((batch for part in parts for batch in part['batches']),
                             key=lambda b:order[b['sample_ids'][0]])
            data = {'context':context, 'videos':videos, 'summary':summarize(videos, initial), 'batches':batches}
            if 'probe' in job:
                data['diagnostic_records'] = sorted(
                    (record for part in parts for record in part['diagnostic_records']),
                    key=lambda r:(order[r['sample_id']], r['step']))
                data['saved_tensor_bytes'] = sum(part['saved_tensor_bytes'] for part in parts)
            if not job.get('partial'):
                write_json(output/'results.json', data)
            yield job, data, None

    def close(self):
        self.close_workers()
        super().close()


def make_evaluator(config):
    gpus = config.get('evaluation_gpus', [0, 1])
    if len(gpus) > 1:
        if len(set(gpus)) != len(gpus) or any(type(g) is not int or g < 0 for g in gpus):
            raise ValueError('evaluation_gpus must contain distinct nonnegative GPU indices')
        return ParallelEvaluator(config['generation'], config['batch_size'], config['state_tensor_bytes'],
            gpus, config.get('worker_log_dir', str(Path(config['output_dir'])/'gpu_workers')))
    return Evaluator(config['generation'], config['batch_size'], config['state_tensor_bytes'])


def _evaluate_jobs(runner, jobs, rows, reference):
    if isinstance(runner, ParallelEvaluator):
        yield from runner.run_jobs(jobs, rows, reference)
    else:
        for job in jobs:
            try:
                data = runner.run(rows, job['policy'], job['output'], reference=reference,
                    shared_context=job.get('context'))
                error = None
            except Exception as exc:
                data, error = None, f'{type(exc).__name__}: {str(exc)[-2000:]}'
            yield job, data, error


def evaluate_jobs(runner, jobs, rows, reference):
    rows = list(rows)
    for job in jobs:
        if not job.get('screen') or len(rows) <= 20 or job.get('initial') or 'probe' in job:
            yield from _evaluate_jobs(runner, [job], rows, reference)
            continue
        output = Path(job['output'])
        context = runner.context(rows, job['policy'], False, reference)
        if (output/'results.json').exists():
            yield job, valid_result(output/'results.json', context), None
            continue
        partial = {**job, 'context':context, 'partial':True}
        _, first, error = next(_evaluate_jobs(runner, [partial], rows[:20], reference))
        if error is not None:
            yield job, None, error
            continue
        scores = first['summary']
        stopped = (scores['mean_psnr'] is not None and scores['mean_psnr'] < 19
                   and scores['mean_ssim'] < 0.65 and scores['mean_lpips'] > 0.25)
        first['screening'] = {
            'status':'early_stopped' if stopped else 'passed',
            'evaluated_samples':20, 'total_samples':len(rows),
            'scope':'Fixed first 20 development samples in configuration order; partial means only.',
            'sample_ids':[r['id'] for r in rows[:20]],
            'rule':'Stop only if mean PSNR < 19 AND mean SSIM < 0.65 AND mean LPIPS > 0.25.',
            'reason':'All three poor-quality screening conditions hold.' if stopped else
                     'The three poor-quality screening conditions do not all hold.'}
        write_json(output/'screening.json', first)
        write_json(output/'progress.json', {'completed_videos':20, 'total_videos':len(rows)})
        if stopped:
            print('Early stopped', job['id'], 'after 20 scored development videos', flush=True)
            yield job, first, None
            continue
        _, rest, error = next(_evaluate_jobs(runner, [{**partial, 'completed_count':20}], rows[20:], reference))
        if error is not None:
            yield job, None, error
            continue
        videos = first['videos'] + rest['videos']
        data = {'context':context, 'videos':videos, 'summary':summarize(videos),
                'batches':first['batches'] + rest['batches']}
        write_json(output/'progress.json', {'completed_videos':len(rows), 'total_videos':len(rows)})
        write_json(output/'results.json', data)
        yield job, data, None
