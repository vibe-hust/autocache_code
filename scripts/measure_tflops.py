"""Measure mean generation TFLOPs on the committed held-out prompts."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
from types import FunctionType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reactive_samplers.data import read_prompts
from reactive_samplers.report import write_json


EXPERIMENTS = {
    'qwen21': ('qwen21-drawbench30-170', 170),
    'flux3': ('flux3-drawbench30-170', 170),
    'wan21': ('wan21-vbench124-820', 820),
}
METHODS = ('official', 'teacache', 'dicache', 'seacache', 'magcache', 'autocache')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare_policy_counting(runtime):
    # PyTorch's dispatcher imports torch from the calling tensor method's frame.
    def import_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name != 'torch' or fromlist or level:
            raise ImportError(f'Unexpected tensor dispatcher import: {name}')
        return sys.modules['torch']

    extract = runtime.extract
    namespace = dict(extract.__globals__, __builtins__={
        **extract.__builtins__, '__import__': import_torch})
    runtime.extract = FunctionType(extract.__code__, namespace)


def count_generation(render, wan_model=None):
    import torch
    from reactive_samplers.operator_counter import OperatorCounter

    meter = OperatorCounter()
    attention = meter.wan_attention(wan_model) if wan_model is not None else nullcontext()
    with torch.inference_mode(), attention:
        output = meter.run(render)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record = meter.record()
    return output, {key: record[key] for key in (
        'tflops', 'registered_flops', 'external_attention_flops', 'unregistered_operators')}


def image_baseline(model, generation, method, settings):
    if method == 'magcache':
        import torch
        from variants.magcache.run import calibrate, check_events, load_generator

        steps = generation['reference_steps']
        transformer, render, _ = load_generator(model, generation)
        with torch.inference_mode():
            _, events = calibrate(settings, steps, transformer, render)

        def measure_magcache(row):
            events.clear()
            _, record = count_generation(lambda: render(row))
            check_events(events, steps * (2 if model == 'flux3' else 1))
            return record

        return measure_magcache

    if model == 'qwen21':
        from baselines.qwen21_cache.run_panel import cache_for, generate_image, load_pipeline

        cfg = {**generation, 'thresholds': settings['thresholds']}
        pipe = load_pipeline(cfg)

        def measure(row):
            cache = cache_for(pipe, cfg, 'original' if method == 'official' else method)
            with cache if cache is not None else nullcontext():
                return count_generation(lambda: generate_image(pipe, cfg, row))[1]
    else:
        from variants.flux3_baselines.generate import cache_for, generate_image, load_pipeline

        pipe = load_pipeline(generation['model_path'])

        def measure(row):
            with cache_for(pipe, method, settings['thresholds'].get(method), settings):
                return count_generation(lambda: generate_image(pipe, row))[1]
    return measure


def image_policy(model, generation, policy):
    from variants.opal_search.policy import Runtime

    if model == 'qwen21':
        from baselines.qwen21_cache.run_panel import load_pipeline
        from reactive_samplers.qwen_batch import BatchAdapter, sample_batch

        adapter = BatchAdapter(load_pipeline(generation), **{
            key: generation[key] for key in ('guidance_scale', 'height', 'width', 'max_sequence_length')})
    else:
        from variants.opal_search.flux3 import load_adapter, sample_batch

        adapter = load_adapter(generation, 'cuda:0')
    steps = generation['reference_steps']

    def measure(row):
        runtime = Runtime(policy)
        prepare_policy_counting(runtime)

        def render():
            latent, _, _ = sample_batch(adapter, [row], runtime, steps, max_consecutive_skips=steps)
            return adapter.finish_batch(latent)

        return count_generation(render)[1]

    return measure


def wan_baseline(method, settings):
    from variants.wan21_baselines.generate import load_entry, upstream_args

    entry = load_entry(ROOT, method)
    pipeline_type = entry.wan.WanT2V
    pipeline = None
    measured = None

    def load_once(**kwargs):
        nonlocal pipeline
        if pipeline is None:
            pipeline = pipeline_type(**kwargs)

            def profile_generate(*args, **kwargs):
                nonlocal measured

                # TeaCache and SeaCache install their generate method on the class.
                video, measured = count_generation(
                    lambda: type(pipeline).generate(pipeline, *args, **kwargs), pipeline.model)
                return video

            pipeline.generate = profile_generate
        return pipeline

    def measure(row):
        nonlocal measured
        measured = None
        args = upstream_args(entry, row, settings, method, 'unused.mp4')
        with patch.object(entry.wan, 'WanT2V', load_once), patch.object(entry, 'cache_video', return_value=None):
            entry.generate(args)
        if measured is None:
            raise RuntimeError(f'Wan generation was not profiled for {row["id"]}')
        return measured

    return measure


def wan_policy(generation, policy):
    from variants.wan21_opal.evaluate import Evaluator

    evaluator = Evaluator(generation, 1, 64 * 1024 * 1024)
    evaluator.load()

    def measure(row):
        adapter, render = evaluator.prepare_generation(row, policy)
        prepare_policy_counting(adapter.runtime)
        with adapter:
            _, record = count_generation(render, evaluator.pipe.model)
        adapter.validate_complete()
        return record

    return measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=EXPERIMENTS)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('output', type=Path, help='Separate TFLOPs results JSON')
    parser.add_argument('--model-path', type=Path, help='Local model snapshot')
    parser.add_argument('--policy', type=Path, help='Frozen policy directory for autocache')
    args = parser.parse_args()
    if args.model == 'wan21' and args.method == 'magcache':
        parser.error('Wan MagCache is a literature result, with no matching local baseline runner')
    if args.policy is not None and args.method != 'autocache':
        parser.error('--policy applies to autocache only')

    experiment, count = EXPERIMENTS[args.model]
    directory = ROOT / 'experiments' / experiment
    prompts = directory / 'test.jsonl'
    rows = read_prompts(prompts, count)
    generation = read_json(directory / 'generation.json')
    generation['model_path'] = str(args.model_path.resolve() if args.model_path else ROOT / generation['model_path'])
    policy = (args.policy or ROOT / 'policies' / args.model).resolve()
    settings = {}
    if args.method == 'magcache':
        settings = read_json(ROOT / 'experiments' / 'magcache' / f'{args.model}.json')
        settings['calibration_prompt'] = str(ROOT / settings['calibration_prompt'])
    elif args.method != 'autocache':
        baseline_config = ROOT / 'experiments/qwen21-drawbench40-baselines/generation.json'
        settings = read_json(baseline_config if args.model == 'qwen21' else directory / 'baselines.json')

    if args.model == 'wan21':
        generation['official_source'] = str(ROOT / generation['official_source'])
        generation['dependency_site_packages'] = str(ROOT / generation['dependency_site_packages'])
        settings['weights'] = generation['model_path']
        measure = wan_policy(generation, policy) if args.method == 'autocache' else wan_baseline(args.method, settings)
    elif args.method == 'autocache':
        measure = image_policy(args.model, generation, policy)
    else:
        measure = image_baseline(args.model, generation, args.method, settings)

    samples = []
    for row in rows:
        record = measure(row)
        samples.append({'id': row['id'], 'seed': row['seed'], **record})
        print(f'{args.model} {args.method} {len(samples)}/{count} {row["id"]}: {record["tflops"]:.4f} TFLOPs', flush=True)
    mean = statistics.mean(row['tflops'] for row in samples)
    write_json(args.output, {
        'model': args.model, 'method': args.method,
        'manifest': str(prompts.relative_to(ROOT)),
        'policy': str(policy) if args.method == 'autocache' else None,
        'summary': {'count': len(samples), 'mean_tflops': mean},
        'samples': samples,
    })
    print(f'Mean: {mean:.4f} TFLOPs over {count} samples', flush=True)


if __name__ == '__main__':
    main()
