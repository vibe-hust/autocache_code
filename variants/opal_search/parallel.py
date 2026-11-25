"""Evaluate different strategies concurrently, one serial evaluator per GPU."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing
import os
import subprocess
from pathlib import Path
import sys

from reactive_samplers.report import write_json
from .evaluate import Evaluator


def gpu_available(gpu, own_pids=()):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[gpu]
    try:
        result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        pids = {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
        if pids - set(own_pids):
            return False
        if pids:
            return True
        result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
        memory, utilization = (int(value.strip()) for value in result.stdout.strip().split(','))
        return memory < 1024 and utilization <= 5
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _initialize(gpus, generation, batch_size, state_bytes, worker_dir):
    global _evaluator, _gpu, _status
    _gpu = gpus.get()
    allocated = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = allocated.split(',')[_gpu] if allocated else str(_gpu)
    folder = Path(worker_dir)
    folder.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = (folder/f'gpu{_gpu}.log').open('a', buffering=1)
    _status = folder/f'gpu{_gpu}.json'
    _evaluator = Evaluator(generation, batch_size, state_bytes)
    _evaluator.device = 'cuda:0'
    write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'ready'})


def _evaluate(job, rows, reference):
    write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'evaluation', 'strategy':job['id']})
    try:
        data = _evaluator.run(rows, job['policy'], job['output'], reference=reference, initial=job.get('initial', False))
        write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'completed', 'strategy':job['id']})
        error = None
    except Exception as exc:
        error = f'{type(exc).__name__}: {str(exc)[-2000:]}'
        write_json(_status, {'gpu':_gpu, 'pid':os.getpid(), 'stage':'failed', 'strategy':job['id'], 'error':error})
        data = None
    pids = [os.getpid()]
    if _evaluator.metrics is not None:
        pids.append(_evaluator.metrics.process.pid)
    return data, error, pids


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
        # Reference generation and optional diagnostics keep their existing serial path.
        self.close_workers()
        return super().run(*args, **kwargs)

    def run_jobs(self, jobs, rows, reference):
        if not jobs:
            return
        slurm = os.environ.get('OPAL_EVALUATION_BACKEND') == 'slurm'
        selected = list(self.gpus) if slurm else self.gpus[:2] if gpu_available(self.gpus[1], self.worker_pids) else [self.gpus[0]]
        if self.pool is not None and selected != self.active_gpus:
            self.close_workers()
        write_json(Path(self.worker_dir)/'dispatch.json', {'gpus':selected,
            'strategies':[job['id'] for job in jobs], 'rule':'Use the Slurm-reserved GPUs' if slurm else
            'Secondary GPU only when free of other compute processes'})
        print('Evaluating strategies on GPUs', selected, flush=True)
        if self.pool is None:
            super().close()
            self.active_gpus = selected
            if slurm:
                self.pool = ThreadPoolExecutor(max_workers=len(selected))
            else:
                context = multiprocessing.get_context('spawn')
                self.gpu_queue = context.Queue()
                for gpu in selected: self.gpu_queue.put(gpu)
                self.pool = ProcessPoolExecutor(max_workers=len(selected), mp_context=context,
                    initializer=_initialize, initargs=(self.gpu_queue, self.config, self.batch_size,
                        self.state_bytes, self.worker_dir))
        worker = _evaluate
        if slurm:
            from .slurm_parallel import evaluate
            worker = partial(evaluate, self.config, self.batch_size, self.state_bytes,
                self.worker_dir)
        futures = {self.pool.submit(worker, job, rows, reference):job for job in jobs}
        for future in as_completed(futures):
            data, error, pids = future.result()
            self.worker_pids.update(pids)
            yield futures[future], data, error

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


def evaluate_jobs(runner, jobs, rows, reference):
    if isinstance(runner, ParallelEvaluator):
        yield from runner.run_jobs(jobs, rows, reference)
    else:
        for job in jobs:
            try:
                data = runner.run(rows, job['policy'], job['output'], reference=reference)
                error = None
            except Exception as exc:
                data, error = None, f'{type(exc).__name__}: {str(exc)[-2000:]}'
            yield job, data, error
