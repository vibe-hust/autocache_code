"""Uncapped loss, seven-objective Pareto selection, and paired feedback."""
import math
import random
import statistics

METRICS = ('clip_score', 'image_reward', 'psnr', 'ssim', 'lpips')
DIRECTIONS = {'mean_clip_score': 1, 'mean_image_reward': 1, 'mean_psnr': 1,
              'mean_ssim': 1, 'mean_lpips': -1, 'mean_nfe': -1, 'mean_seconds': -1}
LOSS_FORMULA = 'sqrt(NFE)/(CLIPScore+100*ImageReward+PSNR+100*SSIM+100-100*LPIPS)'
QUALITY_GATE = {'aggregation':'development-set means', 'combine':'all',
                'clip_score':{'operator':'>','threshold':30.7619},
                'image_reward':{'operator':'>','threshold':0.5325},
                'psnr':{'operator':'>','threshold':29.9745},
                'ssim':{'operator':'>','threshold':0.9186},
                'lpips':{'operator':'<','threshold':0.0918},
                'nfe':{'operator':'<=','threshold':12.7}}


DEFAULT_TARGETS = {key: rule['threshold'] for key, rule in QUALITY_GATE.items()
                   if isinstance(rule, dict) and 'threshold' in rule}
FLUX_TARGETS = dict(psnr=31.9155, ssim=.9700, lpips=.0208, nfe=50)


def configure(targets):
    QUALITY_GATE.clear()
    QUALITY_GATE.update(aggregation='development-set means', combine='all')
    for key, value in targets.items():
        QUALITY_GATE[key] = {'operator': '<' if key == 'lpips' else '<=' if key == 'nfe' else '>',
                             'threshold': value}


def target_description():
    names = dict(clip_score='CLIPScore', image_reward='ImageReward', psnr='PSNR',
                 ssim='SSIM', lpips='LPIPS', nfe='NFE')
    rules = [f"mean {names[key]} {rule['operator']} {rule['threshold']}"
             for key, rule in QUALITY_GATE.items() if key in names]
    if not rules:
        return ('Reduce full backbone evaluations and generation time while seeking better '
                'CLIPScore, ImageReward, PSNR, SSIM and LPIPS on the development set. '
                'There are no fixed reference scores or qualification conditions.')
    return ('Reference targets and final qualification use the same boundaries: ' + '; '.join(rules)
            + '. All conditions must hold. Nonqualifying candidates remain valid search feedback. '
            'Select representatives from qualifying candidates if any exist, otherwise from all '
            'completed new candidates. Balanced minimizes the mean per-sample ranking objective.')


def quality_report(summary):
    criteria = {}
    for key in (*METRICS, 'nfe'):
        if key not in QUALITY_GATE: continue
        value = summary['mean_'+key]
        if key == 'psnr' and value is None: value = math.inf
        rule = QUALITY_GATE[key]; threshold = rule['threshold']
        if rule['operator'] == '>': passed = value > threshold
        elif rule['operator'] == '<': passed = value < threshold
        else: passed = value <= threshold
        gap = max(0., threshold-value) if rule['operator'] == '>' else max(0., value-threshold)
        criteria[key] = {'passed':passed, 'gap_to_boundary':gap, 'at_boundary':value==threshold}
    return {'eligible':all(x['passed'] for x in criteria.values()), 'criteria':criteria}


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


def image_loss(row):
    quality(row)
    s = row['scores']
    if s['psnr'] is None:
        return 0.0
    denominator = s['clip_score'] + 100*s['image_reward'] + s['psnr'] + 100*s['ssim'] + 100 - 100*s['lpips']
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    return math.sqrt(row['nfe'])/denominator


def summarize(rows, initial=False):
    if not rows:
        raise ValueError('Empty evaluation')
    for row in rows:
        quality(row)
        row['loss'] = 1.0 if initial else image_loss(row)
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
    result['images_per_second'] = len(rows)/sum(r['seconds'] for r in rows) if sum(r['seconds'] for r in rows) else None
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
    'best_prompt_alignment': ('mean_clip_score', -1),
    'best_preference_score': ('mean_image_reward', -1),
    'best_reference_fidelity': ('mean_lpips', 1),
    'lowest_nfe': ('mean_nfe', 1),
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
