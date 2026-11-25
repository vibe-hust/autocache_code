"""Uncapped loss, five-objective Pareto selection, and paired feedback."""
import math
import random
import statistics

METRICS = ('psnr', 'ssim', 'lpips')
DIRECTIONS = {'mean_psnr': 1,
              'mean_ssim': 1, 'mean_lpips': -1, 'mean_nfe': -1, 'mean_seconds': -1}
LOSS_FORMULA = 'sqrt(NFE)/(PSNR+100*SSIM+100-100*LPIPS)'
QUALITY_GATE = {'aggregation':'development-set means', 'combine':'all',
                'psnr':{'operator':'>','threshold':25.1022},
                'ssim':{'operator':'>','threshold':0.8520},
                'lpips':{'operator':'<','threshold':0.0904},
                'nfe':{'operator':'<=','threshold':44.0}, 'nfe_required':True}


def quality_report(summary):
    criteria = {}
    for key in ('psnr','ssim','lpips'):
        value = summary['mean_'+key]
        if key == 'psnr' and value is None: value = math.inf
        rule = QUALITY_GATE[key]; threshold = rule['threshold']
        passed = value > threshold if rule['operator'] == '>' else value < threshold
        gap = max(0., threshold-value) if rule['operator'] == '>' else max(0., value-threshold)
        criteria[key] = {'passed':passed, 'gap_to_boundary':gap, 'at_boundary':value==threshold}
    quality_passed = all(x['passed'] for x in criteria.values())
    nfe = summary['mean_nfe']
    if type(nfe) not in (int, float) or not math.isfinite(nfe) or nfe <= 0:
        raise ValueError('Invalid mean NFE')
    threshold = QUALITY_GATE['nfe']['threshold']
    criteria['nfe'] = {'passed': nfe <= threshold, 'gap_to_boundary': max(0., nfe - threshold),
                       'at_boundary': nfe == threshold}
    return {'eligible': quality_passed and criteria['nfe']['passed'],
            'quality_passed': quality_passed, 'criteria': criteria}


def quality(row):
    scores = row['scores']
    for key in METRICS:
        value = scores[key]
        if key == 'psnr' and value is None:
            continue  # Recorded +infinity, never missing data.
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError('Invalid quality metric: '+key)
    if type(row['nfe']) not in (int, float) or not math.isfinite(row['nfe']) or row['nfe'] <= 0:
        raise ValueError('Invalid NFE')


def video_loss(row):
    quality(row)
    s = row['scores']
    if s['psnr'] is None:
        return 0.0
    denominator = s['psnr'] + 100*s['ssim'] + 100 - 100*s['lpips']
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    return math.sqrt(row['nfe'])/denominator


def summarize(rows, initial=False):
    if not rows:
        raise ValueError('Empty evaluation')
    for row in rows:
        quality(row)
        row['loss'] = 1.0 if initial else video_loss(row)
        if not math.isfinite(row['seconds']) or row['seconds'] < 0:
            raise ValueError('Invalid generation timing')
    valid_losses = [r['loss'] for r in rows if r['loss'] is not None]
    result = {'count': len(rows), 'mean_loss': statistics.mean(valid_losses) if valid_losses else None,
              'valid_loss_count': len(valid_losses), 'undefined_loss_count': len(rows)-len(valid_losses),
              'mean_nfe': statistics.mean(r['nfe'] for r in rows),
              'mean_seconds': statistics.mean(r['seconds'] for r in rows),
              'loss_source': 'initial_override_1' if initial else 'formula',
              'seconds_min': min(r['seconds'] for r in rows),
              'seconds_max': max(r['seconds'] for r in rows)}
    for key in METRICS:
        values = [r['scores'][key] for r in rows]
        result['mean_'+key] = None if key == 'psnr' and None in values else statistics.mean(values)
    result['videos_per_second'] = len(rows)/sum(r['seconds'] for r in rows) if sum(r['seconds'] for r in rows) else None
    return result


def objective(summary, key):
    value = summary[key]
    if key == 'mean_psnr' and value is None:
        return math.inf
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Invalid Pareto objective: '+key)
    return value*DIRECTIONS[key]


def pareto(records):
    valid = [r for r in records if r['status'] == 'completed']
    vectors = {r['id']: tuple(objective(r['summary'], k) for k in DIRECTIONS) for r in valid}
    def dominates(a, b):
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))
    return [r['id'] for r in valid if not any(dominates(vectors[o['id']], vectors[r['id']]) for o in valid if o['id'] != r['id'])]


REPRESENTATIVE_METRICS = {
    'best_reference_fidelity': ('mean_lpips', 1),
    'fastest_qualifying': ('mean_seconds', 1),
    'balanced': ('mean_loss', 1),
}


def select_representatives(records):
    completed = [r for r in records if r['status'] == 'completed' and r['id'] != 'initial']
    valid = [r for r in completed if quality_report(r['summary'])['eligible']] or completed
    if not valid:
        return {}
    return {category: min(pool, key=lambda r: (direction*r['summary'][metric],
                r['summary']['mean_nfe'], r['summary']['mean_seconds'], r['id']))
            for category, (metric, direction) in REPRESENTATIVE_METRICS.items()
            if (pool := [r for r in valid if r['summary'][metric] is not None])}


def paired(rows, reference):
    """Signed metric deltas and directional wins; finite paired bootstrap only."""
    other = {(r['id'], r['prompt'], r['seed']): r for r in reference}
    if len(other) != len(rows):
        raise ValueError('Paired sample counts differ')
    output = {}
    for key in (*METRICS, 'nfe', 'seconds', 'loss'):
        deltas = []; wins = losses = ties = infinite = undefined = 0
        direction = -1 if key in ('lpips', 'nfe', 'seconds', 'loss') else 1
        for row in rows:
            ref = other[(row['id'], row['prompt'], row['seed'])]
            a, b = (row['scores'][key], ref['scores'][key]) if key in METRICS else (row[key], ref[key])
            if key == 'loss' and (a is None or b is None):
                undefined += 1; continue
            a = math.inf if key == 'psnr' and a is None else a
            b = math.inf if key == 'psnr' and b is None else b
            if a == b:
                ties += 1; deltas.append(0.0)
            else:
                wins += int((a-b)*direction > 0); losses += int((a-b)*direction < 0)
                if math.isfinite(a-b): deltas.append(a-b)
                else: infinite += 1
        result = {'wins': wins, 'losses': losses, 'ties': ties, 'nonfinite_differences': infinite, 'undefined_pairs': undefined}
        if not infinite and deltas:
            rng = random.Random(0)
            estimates = sorted(sum(rng.choices(deltas, k=len(deltas)))/len(deltas) for _ in range(1000))
            result.update(mean_delta=statistics.mean(deltas), paired_bootstrap_95=[estimates[24], estimates[974]])
        else:
            result.update(mean_delta=None, paired_bootstrap_95=None,
                          note='Infinite PSNR differences: use win/loss counts, not finite confidence intervals.')
        output[key] = result
    return output
