"""Paired fidelity on all 800 images; generation timing is read from saved records."""
import argparse
import json
import os
from pathlib import Path
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=['frozen','seacache','dicache'])
    parser.add_argument('--config',default='experiments/geneval2/generation.json',type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    from reactive_samplers.evaluation import QualityMetrics
    from reactive_samplers.report import write_json
    metrics=QualityMetrics(['psnr','ssim','lpips'])
    rank,workers=int(os.environ.get('SLURM_PROCID',0)),int(os.environ.get('SLURM_NTASKS',1))
    root=Path(cfg['output_dir'])
    rows=json.loads(Path(cfg['prompts']).read_text())
    records=[]
    for row in rows[rank::workers]:
        image=root/args.method/(row['id']+'.png')
        record=json.loads(image.with_suffix('.json').read_text())
        record['scores']=metrics.score(image,row['prompt'],root/'official'/image.name)
        records.append(record)
    write_json(root/args.method/f'scores-{rank}.json',records)
    if workers == 1:
        summary={k:statistics.mean(r[k] for r in records) for k in ('nfe','seconds')}
        for key in ('psnr','ssim','lpips'):
            values=[r['scores'][key] for r in records]
            summary[key]=None if None in values else statistics.mean(values)
        write_json(root/args.method/'results.json',{'count':len(records),'images':records,'summary':summary})


if __name__ == '__main__': main()
