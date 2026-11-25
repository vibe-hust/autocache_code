"""Report fixed development/held-out means using the same matched reference."""
import json
from pathlib import Path
import statistics
import sys


def summarize(root):
    root = Path(root)
    data = json.loads((root/'results.json').read_text())['images']
    data += json.loads((root/'teacache/results.json').read_text())['images']
    cfg = json.loads((root/'config.json').read_text())
    report = {}
    for split in ('development30', 'evaluation170', 'all200'):
        ids = {r['id'] for r in cfg['rows'] if split == 'all200' or r['split'] == split}
        rows = [r for r in data if r['id'] in ids]
        reference = statistics.mean(r['seconds'] for r in rows if r['method'] == 'original')
        report[split] = {}
        for method in dict.fromkeys(r['method'] for r in rows):
            group = [r for r in rows if r['method'] == method]
            result = {'count': len(group)}
            for key in ('nfe', 'seconds'):
                result['mean_'+key] = statistics.mean(r[key] for r in group)
            for key in ('psnr', 'ssim', 'lpips', 'clip_score', 'image_reward'):
                values = [r.get('scores', {}).get(key) for r in group]
                result['mean_'+key] = None if None in values else statistics.mean(values)
            result['speedup'] = reference/result['mean_seconds']
            report[split][method] = result
    (root/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(summarize(sys.argv[1]), indent=2))
