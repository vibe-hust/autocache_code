"""Score complete FLUX DrawBench methods on four GPUs, polling every 10 minutes."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from reactive_samplers.data import drawbench_splits

METHODS = ('official', 'teacache', 'dicache', 'seacache')
NAMES = ('clip_score', 'image_reward', 'psnr', 'ssim', 'lpips')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def ready(run, method, rows):
    folder = run / method
    return all((folder / (row['id'] + suffix)).is_file()
               for row in rows for suffix in ('.png', '.json'))


def paired_record(run, method, row):
    path = run / method / (row['id'] + '.png')
    reference = run / 'official' / path.name
    record = json.loads(path.with_suffix('.json').read_text())
    baseline = json.loads(reference.with_suffix('.json').read_text())
    if record['method'] != method or baseline['method'] != 'official':
        raise ValueError(f'Unexpected image method: {path}')
    if any(record[k] != row[k] or baseline[k] != row[k] for k in ('id', 'prompt', 'seed')):
        raise ValueError(f'Prompt/seed pairing mismatch: {path}')
    if any(record[k] != baseline[k] for k in ('caption', 'steps', 'cfg')):
        raise ValueError(f'Generation settings mismatch: {path}')
    return path, reference, record, baseline


def worker(run, method, rows, rank):
    metrics = None
    for row in rows[rank::4]:
        output = run / 'metrics' / method / (row['id'] + '.json')
        if output.exists():
            continue
        path, reference, record, baseline = paired_record(run, method, row)
        if metrics is None:
            from reactive_samplers.evaluation import QualityMetrics
            weights = run.parents[1] / 'metrics'
            metrics = QualityMetrics(NAMES, weights / 'clip/checkpoints/ViT-B-32.pt',
                weights / 'image_reward/checkpoints/ImageReward.pt',
                weights / 'image_reward/checkpoints/med_config.json')
        values = metrics.score(path, row['prompt'], reference)
        if values['psnr'] is None:
            values['psnr'] = 'Infinity'
        save(output, {**row, 'method': method, 'image': str(path),
                      'reference': str(reference), 'scores': values,
                      'nfe': record['full_cfg_branch_calls']})
        print('SCORED', rank, method, row['id'], flush=True)


def summarize(results):
    summary = dict(count=len(results), nfe=statistics.mean(r['nfe'] for r in results))
    for key in NAMES:
        summary[key] = ('Infinity' if any(r['scores'][key] == 'Infinity' for r in results)
                        else statistics.mean(r['scores'][key] for r in results))
    return summary


def aggregate(run, method, rows):
    folder = run / 'metrics' / method
    results = [json.loads((folder / (row['id'] + '.json')).read_text()) for row in rows]
    summary = summarize(results)
    prompt_dir = Path(__file__).resolve().parents[2] / 'experiments/flux3-drawbench30-170'
    summary['splits'] = {
        name: summarize(group) for name, group in drawbench_splits(results, prompt_dir).items()}
    save(folder / 'results.json', results)
    save(folder / 'summary.json', summary)
    save(folder / 'status.json', dict(status='completed', completed=len(results), total=len(rows)))
    return summary


def score_method(run, method, rows):
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(devices) != 4:
        raise ValueError('Scoring requires the four GPUs assigned to this generation job')
    folder = run / 'metrics' / method
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'status.json', dict(status='scoring', total=len(rows)))
    with ExitStack() as stack:
        processes = []
        for rank, device in enumerate(devices):
            log = stack.enter_context((folder / f'worker-{rank}.log').open('a'))
            processes.append(subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), str(run),
                 '--method', method, '--rank', str(rank)],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': device}, stdout=log, stderr=log))
        codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f'{method} scoring failed: {codes}; see {folder}/worker-*.log')
    return aggregate(run, method, rows)


def watch(run, rows):
    output = run / 'metrics'
    save(output / 'protocol.json', dict(
        reference='Same prompt/seed official image; original DrawBench prompts',
        quality='Existing QualityMetrics: CLIPScore, ImageReward, PSNR, SSIM and LPIPS; no loss',
        nfe='Original full_cfg_branch_calls; CFG branches counted separately; partial probes excluded',
        poll_seconds=600, workers=4))
    summary = {}
    for method in METHODS:
        path = output / method / 'status.json'
        if path.exists() and json.loads(path.read_text())['status'] == 'completed':
            summary[method] = aggregate(run, method, rows)
    while len(summary) != len(METHODS):
        if ready(run, 'official', rows):
            for method in METHODS:
                if method not in summary and ready(run, method, rows):
                    save(output / 'status.json', dict(status='scoring', method=method,
                                                     completed_methods=list(summary)))
                    summary[method] = score_method(run, method, rows)
                    save(output / 'summary.json', summary)
                    print('SCORED_METHOD', method, summary[method], flush=True)
        pending = [method for method in METHODS if method not in summary]
        save(output / 'status.json', dict(status='waiting' if pending else 'completed',
                                         completed_methods=list(summary), pending_methods=pending))
        if pending:
            print('WAITING_FOR_GENERATION', pending, 'next check in 600 seconds', flush=True)
            time.sleep(600)
    save(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--rank', type=int, choices=range(4), default=0)
    args = parser.parse_args()
    rows = json.loads((args.run / 'prompts.json').read_text())
    if len(rows) != 200 or len({r['id'] for r in rows}) != 200:
        raise ValueError('Expected the fixed 200-image panel with unique IDs')
    if args.method:
        worker(args.run, args.method, rows, args.rank)
    else:
        watch(args.run, rows)


if __name__ == '__main__':
    main()
