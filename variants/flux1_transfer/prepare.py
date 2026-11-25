"""Prepare the recorded 50-step FLUX.1 transfer panel from committed inputs."""
import json
from pathlib import Path
import sys

from reactive_samplers.report import write_json


def prepare(profile):
    cfg = json.loads(Path(profile).read_text())
    cfg['model'] = str(Path(cfg['model']).resolve())
    cfg['checkpoints'] = {k: str(Path(v).resolve()) for k, v in cfg['checkpoints'].items()}
    root = Path(cfg.pop('output_dir')).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cfg['rows'] = [dict(json.loads(line), split=split)
                   for split, key in [('development30', 'development_prompts'), ('evaluation170', 'test_prompts')]
                   for line in Path(cfg.pop(key)).read_text().splitlines() if line.strip()]
    source = Path(cfg.pop('policy'))
    (root/'policy').mkdir()
    for name in ('signals.py', 'updater.py'):
        (root/'policy'/name).write_bytes((source/name).read_bytes())
    for method in ('original', cfg['policy_name'], 'dicache', 'seacache'):
        (root/method).mkdir()
        (root/'trajectories'/method).mkdir(parents=True)
    write_json(root/'config.json', cfg)
    tea = root/'teacache'
    (tea/'images').mkdir(parents=True)
    write_json(tea/'config.json', {**cfg, 'threshold': cfg['teacache_threshold'],
        'reference_dir': str(root/'original'), 'reference_results': str(root/'results.json')})
    print(root)
    return root


if __name__ == '__main__':
    prepare(sys.argv[1])
