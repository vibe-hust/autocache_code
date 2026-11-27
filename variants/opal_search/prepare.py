"""Prepare image search and evaluation from scored all-NEW reference images."""
import argparse
from pathlib import Path
from struct import unpack
from tempfile import TemporaryDirectory

from reactive_samplers.report import write_json
from .evaluate import Evaluator, read
from .search import load_config, reference_directory
from .selection import summarize


def reference_images(sources, generation, path_map):
    model = generation['model']
    steps = generation['reference_steps']
    expected_nfe = steps * (2 if model == 'flux3' else 1)
    records = {}
    for source in sources:
        data = read(source)
        images = data['images'] if isinstance(data, dict) else data
        if model == 'qwen21':
            if 'context' in data:
                if not data['context']['initial']:
                    raise ValueError(f'Expected an all-NEW reference: {source}')
                settings = data['context']['generation']
            else:
                settings = read(source.parents[1]/'config.json')
            keys = ('model', 'dtype', 'reference_steps', 'guidance_scale', 'height', 'width', 'max_sequence_length')
            if any(settings[key] != generation[key] for key in keys):
                raise ValueError(f'Qwen reference generation settings differ: {source}')
        for record in images:
            image = Path(record['image'])
            if path_map and image.is_relative_to(path_map[0]):
                image = path_map[1] / image.relative_to(path_map[0])
            image = image.resolve()
            with image.open('rb') as stream:
                header = stream.read(24)
            if not header.startswith(b'\x89PNG\r\n\x1a\n') or len(header) != 24:
                raise ValueError(f'Expected a PNG reference image: {image}')
            if unpack('>II', header[16:24]) != (generation['width'], generation['height']):
                raise ValueError(f'Reference image size differs: {image}')
            if model == 'flux3':
                meta = read(image.with_suffix('.json'))
                if (meta['method'] != 'official' or meta['caption'] != record['prompt']
                        or meta['steps'] != steps or meta['cfg'] != generation['guidance_scale']
                        or meta['full_cfg_branch_calls'] != expected_nfe
                        or any(meta[k] != record[k] for k in ('id', 'prompt', 'seed'))):
                    raise ValueError(f'FLUX3 reference generation settings differ: {image}')
                seconds = meta['seconds']
            else:
                if record.get('method', 'original') != 'original':
                    raise ValueError(f'Expected an original Qwen image: {image}')
                seconds = record['seconds']
            key = (record['id'], record['prompt'], record['seed'])
            if record['nfe'] != expected_nfe or key in records:
                raise ValueError(f'Expected one all-NEW reference per sample: {key[0]}')
            scores = dict(record['scores'], psnr=None, ssim=1.0, lpips=0.0)
            records[key] = dict(image=str(image), scores=scores, nfe=expected_nfe, seconds=seconds)
    return records


def prepare(profile, sources, destination=None, path_map=None):
    config = load_config(profile)
    model = config['generation']['model']
    if model not in ('qwen21', 'flux3') or config['batch_size'] != 1:
        raise ValueError('Reference import supports one-sample Qwen and FLUX3 DrawBench runs')
    run = Path(config['output_dir'])
    if run.exists():
        raise FileExistsError(f'Choose a fresh search output directory: {run}')
    destination = Path(destination or reference_directory(run)).resolve()
    if destination.is_relative_to(run) or destination.exists():
        raise ValueError(f'Choose a fresh reference directory outside the search output: {destination}')
    images = reference_images(sources, config['generation'], path_map)
    steps = config['generation']['reference_steps']
    branches = 2 if model == 'flux3' else 1
    evaluator = Evaluator(config['generation'], config['batch_size'], config['state_tensor_bytes'])
    groups = {'development': config['rows'], 'test': config['test_rows']}
    if set(images) != {(r['id'], r['prompt'], r['seed']) for rows in groups.values() for r in rows}:
        raise ValueError('Reference results must match both prompt-and-seed lists')

    prepared = read(profile)
    prepared['reference_results'] = str(destination/'development/results.json')
    prepared['test_reference_results'] = str(destination/'test/results.json')
    trace = ''.join(f'{{"step":{i},"reuse_cache":false,"nfe":{(i+1)*branches}}}\n' for i in range(steps))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary)/'references'
        for split, rows in groups.items():
            folder = staged/split
            folder.mkdir(parents=True)
            records = []
            for row in rows:
                name = row['id'] + '.jsonl'
                (folder/name).write_text(trace, encoding='utf-8')
                records.append({**row, **images[row['id'], row['prompt'], row['seed']],
                    'trajectory': str(destination/split/name), 'observations_available': False})
            write_json(folder/'results.json', {
                'context': evaluator.context(rows, None, True, None),
                'images': records, 'summary': summarize(records, initial=True), 'batches': [],
                'provenance': {'source_results': [str(p.resolve()) for p in sources]},
            })
        write_json(staged/'search.json', prepared)
        staged.rename(destination)
    return destination/'search.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('references', type=Path, nargs='+', help='Scored original result JSON files covering the full panel')
    parser.add_argument('--output', type=Path, help='Directory for the prepared configuration and reference indices')
    parser.add_argument('--path-map', type=Path, nargs=2, metavar=('OLD', 'NEW'), help='Replace a prefix in archived image paths')
    args = parser.parse_args()
    print(prepare(args.config, args.references, args.output, args.path_map))
