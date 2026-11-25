"""Select a small 20-step rule adjustment on development30, then freeze/test170."""
import json
from pathlib import Path
import sys
import time

import torch
from diffusers import FluxPipeline

from reactive_samplers.qwen_metrics import LegacyMetrics
from variants.opal_search.flux1 import sample_batch
from variants.opal_search.policy import Runtime
from .run import ECADSettingsAdapter, save, summary


@torch.inference_mode()
def main(root):
    cfg = json.loads((root / 'config.json').read_text())
    reference = {r['id']: r for r in json.loads(Path(cfg['source_results']).read_text())['images']
                 if r['method'] == 'original'}
    pipe = FluxPipeline.from_pretrained(cfg['model'], torch_dtype=torch.bfloat16,
                                        local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    adapter = ECADSettingsAdapter(pipe, guidance_scale=5, height=256, width=256,
                                  max_sequence_length=512)

    def generate(row, policy):
        adapter.prepare_prompt(row)
        runtime = Runtime(policy)
        latent, traces, timing = sample_batch(adapter, [row], runtime, 20, max_consecutive_skips=20)
        return adapter.finish_batch(latent)[0], traces[0], timing['nfe'][0]

    for candidate in cfg['candidates']:
        generate(cfg['rows'][0], candidate['policy'])  # Exclude warm-up calls.
    metrics = LegacyMetrics(cfg['metrics_python'], cfg['metrics'])

    def evaluate(rows, policy, output, stage):
        output.mkdir(parents=True, exist_ok=False)
        records = []
        for row in rows:
            adapter.synchronize()
            began = time.perf_counter()
            image, trace, nfe = generate(row, policy)
            adapter.synchronize()
            seconds = time.perf_counter() - began
            path = output / (row['id'] + '.png')
            image.save(path)
            (output / (row['id'] + '.jsonl')).write_text(
                ''.join(json.dumps(t, allow_nan=False) + '\n' for t in trace))
            records.append(dict(row, image=str(path), nfe=nfe, seconds=seconds,
                                reference_image=reference[row['id']]['image'],
                                scores=metrics.score(path, row['prompt'], reference[row['id']]['image'])))
            result = summarize(records)
            save(output / 'results.json', result)
            save(root / 'status.json', dict(status='running', stage=stage,
                                            completed=len(records), total=len(rows)))
            print(json.dumps(dict(stage=stage, completed=len(records), total=len(rows),
                                  summary=result['summary'])), flush=True)
        return result

    def summarize(records):
        result = summary(records)
        full_seconds = sum(reference[r['id']]['seconds'] for r in records) / len(records)
        result.update(full_reference_mean_seconds=full_seconds, speedup=full_seconds / result['mean_seconds'])
        return dict(summary=result, images=records)

    try:
        candidates = []
        for candidate in cfg['candidates']:
            result = evaluate(cfg['rows'][:30], candidate['policy'],
                              root / 'development' / candidate['name'], candidate['name'] + '/development30')
            candidates.append(dict(candidate, summary=result['summary']))
        selected = min(candidates, key=lambda c: (c['summary']['mean_lpips'], -c['summary']['mean_psnr'],
                                                 c['summary']['mean_seconds'], c['name']))
        save(root / 'selected.json', dict(rule=cfg['selection'], candidates=candidates, selected=selected))
        frozen = root / 'frozen_policy'
        frozen.mkdir()
        for path in Path(selected['policy']).iterdir():
            (frozen / path.name).write_bytes(path.read_bytes())
        Runtime(frozen)
        print('SELECTED', json.dumps(selected), flush=True)
        test = evaluate(cfg['rows'][30:], frozen, root / 'evaluation170', selected['name'] + '/evaluation170')
        dev = json.loads((root / 'development' / selected['name'] / 'results.json').read_text())
        all_rows = dev['images'] + test['images']
        result = summarize(all_rows)
        result.update(policy=selected['name'], selection_rule=cfg['selection'],
                      splits=dict(development30=dev['summary'], evaluation170=test['summary']))
        save(root / 'results.json', result)
        prior = json.loads(Path(cfg['source_comparison']).read_text())
        comparison = dict(original_r09_s03=prior['r09_s03'], ecad_fast=prior['ecad_fast'],
                          adjusted_r09_s03=dict(policy=selected['name'], all200=result['summary'],
                                              development30=dev['summary'], evaluation170=test['summary']))
        save(root / 'comparison.json', comparison)
        save(root / 'status.json', dict(status='completed', selected_policy=selected['name'],
                                        completed=200, total=200, development_candidates=2))
        print('COMPLETED', json.dumps(comparison), flush=True)
    finally:
        metrics.close()


if __name__ == '__main__':
    main(Path(sys.argv[1]))
