"""Create the matched official/BudCache 28-step panel without generating images."""
import json
from pathlib import Path
import sys

from reactive_samplers.report import write_json


def prepare(profile):
    cfg = json.loads(Path(profile).read_text())
    root = Path(cfg.pop('output_dir')).resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = [dict(json.loads(line), split=split)
            for split, path in [('exploration', 'experiments/flux3-drawbench30-170/development.jsonl'),
                                ('evaluation', 'experiments/flux3-drawbench30-170/test.jsonl')]
            for line in Path(path).read_text().splitlines() if line.strip()]
    write_json(root/'config.json', cfg)
    write_json(root/'prompts.json', rows)
    print(root)
    return root


if __name__ == '__main__':
    prepare(sys.argv[1])
