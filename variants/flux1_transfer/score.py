import json
from pathlib import Path
import statistics
import sys

from reactive_samplers.evaluation import QualityMetrics

root = Path(sys.argv[1])
cfg = json.loads((root / 'config.json').read_text())
data = json.loads((root / 'generation_results.json').read_text())
(root / 'status.json').write_text(json.dumps({'status': 'loading_semantic_scorers',
    'completed': 0, 'total': len(data['images'])}))
metrics = QualityMetrics(['clip_score', 'image_reward'], **cfg['checkpoints'])
path = root / 'semantic_scores.jsonl'
records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
done = {(r['method'], r['id']) for r in records}
with path.open('a') as out:
    for task in data['images']:
        if (task['method'], task['id']) in done:
            continue
        row = {'method': task['method'], 'id': task['id'],
               'scores': metrics.score(task['image'], task['prompt'])}
        out.write(json.dumps(row) + '\n')
        out.flush()
        records.append(row)
        if len(records) % 20 == 0:
            (root / 'status.json').write_text(json.dumps({'status': 'semantic_scoring',
                'completed': len(records), 'total': len(data['images'])}))
            print('SCORED', len(records), '/', len(data['images']), flush=True)
lookup = {(r['method'], r['id']): r['scores'] for r in records}
for row in data['images']:
    row['scores'] = {**row.get('scores', {}), **lookup[row['method'], row['id']]}
for method, summary in data['summary'].items():
    group = [r for r in data['images'] if r['method'] == method]
    for key in ['clip_score', 'image_reward']:
        summary['mean_' + key] = statistics.mean(r['scores'][key] for r in group)
(root / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
(root / 'status.json').write_text(json.dumps({'status': 'completed', 'summary': data['summary']}, indent=2) + '\n')
print('COMPLETE', json.dumps(data['summary']), flush=True)
