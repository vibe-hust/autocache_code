"""Prepare the saved 20-step comparison and its two development-only variants."""
import json
from pathlib import Path
import sys
import tempfile

from reactive_samplers.report import write_json
from variants.opal_search.policy import FILES, Runtime, freeze


def prepare(profile):
    cfg = json.loads(Path(profile).read_text())
    root = Path(cfg.pop('output_dir')).resolve()
    adjusted = Path(cfg.pop('adjusted_output_dir')).resolve()
    source = Path(cfg.pop('source_policy'))
    root.mkdir(parents=True, exist_ok=False)
    adjusted.mkdir(parents=True, exist_ok=False)
    cfg['rows'] = [json.loads(line) for key in ('development_prompts', 'test_prompts')
                   for line in Path(cfg.pop(key)).read_text().splitlines() if line.strip()]
    policy = root/'r09_s03'
    policy.mkdir()
    for name in (*FILES, 'manifest.json'):
        (policy/name).write_bytes((source/name).read_bytes())
    freeze(Path(__file__).parents[1]/'opal_search/templates', root/'all_new', 'flux')
    cfg['policy'] = str(policy)
    write_json(root/'config.json', cfg)
    updater = (source/'updater.py').read_text()
    old = '(0, 1, 2, 3, 4, 6, 9, 13,'
    assert updater.count(old) == 3
    candidates = []
    for step, replacement in [(3, '(0, 1, 2, 4, 6, 9, 13,'), (4, '(0, 1, 2, 3, 6, 9, 13,')]:
        name = 'r09_s03_20step_drop'+str(step)
        target = adjusted/'policies'/name
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'signals.py').write_bytes((source/'signals.py').read_bytes())
            (folder/'updater.py').write_text(updater.replace(old, replacement))
            freeze(folder, target, 'flux')
        Runtime(target)
        candidates.append({'name': name, 'policy': str(target), 'removed_refresh_step': step})
    write_json(adjusted/'config.json', {**cfg, 'source_results': str(root/'ours/results.json'),
        'source_comparison': str(root/'comparison.json'), 'candidates': candidates,
        'selection': 'Lowest development-mean LPIPS; ties: higher PSNR, lower latency, then name.'})
    print(json.dumps({'reference_comparison': str(root), 'development_adjustment': str(adjusted)}))
    return root, adjusted


if __name__ == '__main__':
    prepare(sys.argv[1])
