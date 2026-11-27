"""Expose the bundled SeaCache helpers to the Qwen and FLUX.1 adapters."""
from pathlib import Path
from runpy import run_path

_upstream = run_path(str(Path(__file__).resolve().parents[2] / 'baselines/SeaCache/Wan2.1/util_seacache.py'))
apply_sea_with_scheduler = _upstream['apply_sea_with_scheduler']
rel_l1 = _upstream['rel_l1']
