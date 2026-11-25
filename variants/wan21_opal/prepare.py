"""Prepare a new video search from completed Wan baseline references, without generation."""
import argparse
import json
from pathlib import Path

from reactive_samplers.report import write_json
from .evaluate import Evaluator, file_hash, read, validate_generation, validate_rows
from .selection import summarize


def import_reference(baseline, rows, output, generation, state_bytes):
    output.mkdir(parents=True, exist_ok=False)
    manifest = read(baseline / 'manifest.json')
    matched = {r['id']: r for r in manifest['samples']}
    records = []
    for row in rows:
        if any(matched[row['id']][k] != row[k] for k in ('id', 'prompt', 'seed', 'split')):
            raise ValueError('Baseline sample mismatch')
        video = baseline / 'official' / row['split'] / (row['id'] + '.mp4')
        saved = read(video.with_suffix('.json'))
        if any(saved[k] != row[k] for k in ('id', 'prompt', 'seed', 'split')):
            raise ValueError('Reference metadata mismatch')
        if saved['sha256'] != file_hash(video):
            raise ValueError('Corrupt baseline reference')
        steps = generation['reference_steps']
        if saved['nfe'] != 2 * steps or len(saved['full_calls']) != 2 * steps or not all(saved['full_calls']):
            raise ValueError('Reference is not all NEW')
        if saved['frames'] != generation['frames'] or saved['fps'] != generation['fps'] or saved['steps'] != steps:
            raise ValueError('Reference video configuration mismatch')
        trajectory = output / (row['id'] + '.jsonl')
        trajectory.write_text(''.join(json.dumps({'step': step, 'reuse_cache': False,
            'nfe': 2 * (step + 1), 'full_cond': True, 'full_uncond': True,
            'observation': None, 'signals': None, 'source': 'baseline recorded full_calls'}) + '\n'
            for step in range(steps)))
        records.append({**row, 'video': str(video), 'video_sha256': saved['sha256'],
            'trajectory': str(trajectory), 'trajectory_sha256': file_hash(trajectory),
            'observations_available': False, 'scores': {'psnr': None, 'ssim': 1., 'lpips': 0.},
            'nfe': saved['nfe'], 'seconds': saved['generation_seconds'],
            'encode_seconds': saved['encode_seconds'], 'full_prediction_nodes': steps})
    context = Evaluator(generation, 1, state_bytes).context(rows, Path(__file__).parent/'templates', True, None)
    path = output / 'results.json'
    write_json(path, {'context': context, 'videos': records, 'summary': summarize(records, initial=True),
                     'batches': [], 'source_manifest_sha256': file_hash(baseline/'manifest.json'),
                     'note': 'Imported existing reference videos and measured times. Node observations were not recorded.'})
    return str(path)


def prepare(baseline, destination, agent_config, steps=50,
            exploration_prompts=None, evaluation_prompts=None):
    destination = Path(destination).resolve()
    baseline = Path(baseline).resolve() if baseline else None
    if bool(exploration_prompts) != bool(evaluation_prompts):
        raise ValueError('Provide both exploration and evaluation prompt files')
    if baseline:
        manifest = read(baseline/'manifest.json')
        if manifest['steps'] != steps:
            raise ValueError(f"Requested {steps} steps but reference manifest records {manifest['steps']} steps")
    else:
        manifest = {'model': str(Path(__file__).resolve().parents[2]/'models/Wan2.1-T2V-1.3B'),
                    'batch_size': 1, 'height': 480, 'width': 832, 'frames': 81,
                    'fps': 16, 'solver': 'unipc', 'shift': 8, 'cfg': 6,
                    't5_cpu': False, 'offload_model': False}
    if exploration_prompts:
        development = [{**row, 'split': 'search'} for row in read(exploration_prompts)]
        testing = [{**row, 'split': 'test'} for row in read(evaluation_prompts)]
        for group in (development, testing):
            if any(row['sampling_dimension'] not in row['dimensions']
                   for row in group if 'sampling_dimension' in row):
                raise ValueError('Sampling dimension must belong to the prompt dimension labels')
        rows = development + testing
        split_sizes = {'search': len(development), 'test': len(testing)}
    elif baseline:
        panel = read(baseline/'panel30.json')
        rows = panel['samples']
        split_sizes = {key: sum(row['split'] == key for row in rows) for key in ('search', 'test')}
        if split_sizes != panel['split_sizes'] or sum(split_sizes.values()) != len(rows):
            raise ValueError('Panel split mismatch')
        development = [row for row in rows if row['split'] == 'search']
        testing = [row for row in rows if row['split'] == 'test']
    else:
        raise ValueError('Provide exploration/evaluation prompt files or a baseline root')
    validate_rows(rows)
    development_keys = {' '.join(row['prompt'].casefold().split()) for row in development}
    test_keys = {' '.join(row['prompt'].casefold().split()) for row in testing}
    if development_keys & test_keys:
        raise ValueError('Development and test prompts overlap')
    project = Path(__file__).resolve().parents[2]
    upstream = project/'baselines/Wan2.1'
    generation = {'model': 'wan21', 'method': 'residual_cache', 'model_path': manifest['model'],
        'official_source': str(upstream), 'dtype': 'bfloat16', 'batch_size': manifest['batch_size'],
        'dependency_site_packages': str(project),
        'reference_steps': steps, 'height': manifest['height'], 'width': manifest['width'],
        'frames': manifest['frames'], 'fps': manifest['fps'], 'solver': manifest['solver'],
        'shift': manifest['shift'], 'guidance_scale': manifest['cfg'],
        't5_cpu': manifest['t5_cpu'], 'offload_model': manifest['offload_model'],
        'source_sha256': {}}
    for path, digest in manifest.get('code_sha256', {}).items():
        p = Path(path)
        if p.is_relative_to(upstream):
            generation['source_sha256'][str(p.relative_to(upstream))] = digest
    validate_generation(generation)
    agent = read(agent_config)
    if not agent.get('model'):
        raise ValueError('Agent config must explicitly name a model')
    destination.mkdir(parents=True, exist_ok=False)
    state_bytes = 64 * 1024 * 1024
    write_json(destination/'generation.json', generation)
    for filename, data in [('development.jsonl', development), ('test.jsonl', testing)]:
        (destination/filename).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))
    if exploration_prompts:
        (destination/'exploration_prompts.json').write_bytes(Path(exploration_prompts).read_bytes())
        (destination/'evaluation_prompts.json').write_bytes(Path(evaluation_prompts).read_bytes())
    references = {}
    if baseline:
        references['reference_results'] = import_reference(
            baseline, development, destination/'references/development', generation, state_bytes)
        references['test_reference_results'] = import_reference(
            baseline, testing, destination/'references/test', generation, state_bytes)
    config = {'generation_config': str(destination/'generation.json'),
        'development_prompts': str(destination/'development.jsonl'), 'test_prompts': str(destination/'test.jsonl'),
        **references,
        'output_dir': str(destination/'search'), 'batch_size': 1, 'round_sizes': [10]+[5]*8,
        'state_tensor_bytes': state_bytes, 'max_replacement_rounds': 3,
        'diagnostic_budget': 9, 'max_agent_calls_per_round': 12, 'agent': agent,
        'reference_notes': (
            'Existing all-NEW reference videos are imported. ' if baseline else
            'All-NEW references will be generated with the configured prompts, seeds and step count. ')
            + f'NFE counts both CFG branches, so the {steps}-step initial policy has NFE={2 * steps}.'}
    write_json(destination/'search.json', config)
    print(json.dumps({'config': str(destination/'search.json'), 'split_sizes': split_sizes,
                      'gpu_generation': False, 'paid_agent_calls': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, choices=(40, 50), default=50)
    parser.add_argument('--baseline-root')
    parser.add_argument('--exploration-prompts')
    parser.add_argument('--evaluation-prompts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--agent-config', required=True)
    args = parser.parse_args()
    prepare(args.baseline_root, args.output, args.agent_config, args.steps,
            args.exploration_prompts, args.evaluation_prompts)
