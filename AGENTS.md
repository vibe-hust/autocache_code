# AutoCache paper repository

The user is the authority. Keep changes minimal and preserve the official cache,
solver, CFG, precision, prompts and seeds unless the user requests a change.

- Paper image searches: `variants/opal_search/` (Qwen-Image-2.1 and FLUX3 Klein Base).
- Paper video search: `variants/wan21_opal/` (Wan2.1-T2V-1.3B).
- Policies edit `signals.py` and `updater.py`; the framework owns residuals,
  observations, evaluation and final development-only selection.
- Formal data/configurations live in `experiments/`; frozen Single programs in
  `policies/`. Keep DrawBench 30/170, VBench 124/820 and GenEval 2 800 unchanged.
- Shared observation definitions, tracing and manifest reading are under
  `reactive_samplers/`. `models/sd3.py` remains an inherited adapter base used by
  Qwen/FLUX; its filename does not denote a reported SD3 experiment.
- Preserve the FLUX.1 transfer and 20/28-step comparisons under `variants/flux1_*`
  and the shared FLUX.1 search adapter. Do not mistake cited-only Wan baseline
  results for locally generated experiments.
- Keep licenses and runtime dependencies of retained upstream sources.
- Generated images/videos, models, environments, logs and credentials stay out of
  Git. Never delete production experiment directories during repository cleanup.
- Development-only checks remain outside this repository. GPU generation and
  paid Agent calls require an execution request covering that work.
