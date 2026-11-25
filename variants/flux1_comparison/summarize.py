"""Report pooled 200-prompt and fixed 30/170 split means without changing seeds."""
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
config = json.loads((root / 'config.json').read_text())
splits = {'all200': {r['id'] for r in config['rows']},
          'development30': {r['id'] for r in config['rows'][:30]},
          'evaluation170': {r['id'] for r in config['rows'][30:]}}
report = {}
common_reference = json.loads((root / 'ours/results.json').read_text())['images']
for name, method in [('ours', 'r09_s03'), ('ecad', 'ecad_fast')]:
    data = json.loads((root / name / 'results.json').read_text())
    assert len([r for r in data['images'] if r['method'] == method]) == 200
    report[method] = {}
    for split, ids in splits.items():
        rows = [r for r in data['images'] if r['method'] == method and r['id'] in ids]
        reference = [r for r in data['images'] if r['method'] == 'original' and r['id'] in ids]
        original_seconds = statistics.mean(r['seconds'] for r in common_reference
                                           if r['method'] == 'original' and r['id'] in ids)
        seconds = statistics.mean(r['seconds'] for r in rows)
        report[method][split] = dict(count=len(rows), mean_seconds=seconds,
            mean_nfe=statistics.mean(r['nfe'] for r in rows) if 'nfe' in rows[0] else None,
            full_reference_mean_seconds=original_seconds, speedup=original_seconds / seconds,
            paired_implementation_speedup=statistics.mean(r['seconds'] for r in reference) / seconds,
            **{'mean_' + key: statistics.mean(r['scores'][key] for r in rows)
               for key in ('psnr', 'ssim', 'lpips', 'clip_score', 'image_reward')})
report['speedup_reference'] = 'Shared uncached stock Diffusers baseline from ours/results.json.'
(root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
