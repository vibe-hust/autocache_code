"""Run the existing strategy evaluator on one Slurm-assigned GPU."""
import os
from pathlib import Path
import subprocess
import sys

from reactive_samplers.report import write_json
from .evaluate import read


def evaluate(generation, batch_size, state_bytes, worker_dir, job, rows, reference):
    folder = Path(worker_dir)/job['id']
    folder.mkdir(parents=True, exist_ok=True)
    request = folder/'request.json'
    write_json(request, {'generation':generation, 'batch_size':batch_size,
        'state_bytes':state_bytes, 'job':job, 'rows':rows,
        'reference':None if reference is None else str(reference)})
    command = ['srun', '--nodes=1', '--ntasks=1', '--ntasks-per-node=1',
        '--cpus-per-task=20', '--gres=gpu:1', '--gpus-per-task=1',
        '--gpu-bind=single:1', '--exact', '--exclusive', '--kill-on-bad-exit=1',
        sys.executable, '-m', 'variants.opal_search.slurm_parallel', str(request)]
    with (folder/'worker.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(folder/'response.json')


if __name__ == '__main__':
    from queue import SimpleQueue
    import socket
    from . import parallel
    path = Path(sys.argv[1])
    request = read(path)
    print('Slurm strategy worker', socket.gethostname(),
          'CUDA_VISIBLE_DEVICES='+os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
    gpus = SimpleQueue()
    gpus.put(0)  # _initialize preserves the one physical device assigned by Slurm.
    parallel._initialize(gpus, request['generation'], request['batch_size'],
        request['state_bytes'], path.parent/'runtime')
    try:
        result = parallel._evaluate(request['job'], request['rows'], request['reference'])
        write_json(path.with_name('response.json'), result)
    finally:
        parallel._evaluator.close()
