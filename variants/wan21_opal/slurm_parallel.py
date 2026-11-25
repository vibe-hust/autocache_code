"""Run the upstream sample-shard evaluator on one Slurm-assigned GPU."""
import os
from pathlib import Path
import subprocess
import sys

from reactive_samplers.report import write_json
from .evaluate import read


def evaluate(generation, batch_size, state_bytes, worker_dir, job, rows, reference, context):
    folder = Path(worker_dir)/job['id']/rows[0]['id']
    folder.mkdir(parents=True, exist_ok=True)
    request = folder/'request.json'
    write_json(request, {'generation':generation, 'batch_size':batch_size,
        'state_bytes':state_bytes, 'job':job, 'rows':rows,
        'reference':None if reference is None else str(reference), 'context':context})
    command = ['srun', '--nodes=1', '--ntasks=1', '--ntasks-per-node=1',
        '--cpus-per-task=20', '--gres=gpu:1', '--gpus-per-task=1',
        '--gpu-bind=single:1', '--exact', '--exclusive', '--kill-on-bad-exit=1',
        sys.executable, '-m', 'variants.wan21_opal.slurm_parallel', str(request)]
    with (folder/'worker.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(folder/'response.json')


if __name__ == '__main__':
    from queue import SimpleQueue
    import socket
    from . import parallel
    request_path = Path(sys.argv[1])
    request = read(request_path)
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    print('Slurm sample worker', socket.gethostname(), 'CUDA_VISIBLE_DEVICES='+visible, flush=True)
    gpus = SimpleQueue()
    gpus.put(visible)  # Preserve Slurm's assigned device, including physical IDs or UUIDs.
    parallel._initialize(gpus, request['generation'], request['batch_size'],
        request['state_bytes'], request_path.parent/'runtime')
    try:
        result = parallel._evaluate(request['job'], request['rows'],
            request['reference'], request['context'])
        write_json(request_path.with_name('response.json'), result)
    finally:
        parallel._evaluator.close()
