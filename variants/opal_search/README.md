# Image policy search

Qwen-Image-2.1 and FLUX3-Image-Klein-Base share this controller. The Agent edits
`extract_signals(tensors, state)` in `signals.py` and
`choose(observation, signals, state)` in `updater.py`. Cache execution, solver,
CFG, precision, prompt/seed lists and evaluation belong to the framework.

Use the committed DrawBench 30/170 manifests and configurations:

```bash
python -m variants.opal_search.search check experiments/qwen21-drawbench30-170/search.json
python -m variants.opal_search.search run experiments/qwen21-drawbench30-170/search.json
python -m variants.opal_search.search test experiments/qwen21-drawbench30-170/search.json
```

Use `experiments/flux3-drawbench30-170/search.json` for FLUX. `check` is CPU-only.
`run --resume` continues the same recorded experiment. Agent credentials and the
RunPod relay are configured separately, as described in the root README.

Nine rounds offer 10 + 8x5 candidate slots, with bounded replacement submissions.
The controller records failed, duplicate and completed candidates separately.
Default evidence contains the cumulative Pareto frontier and qualifying policies
outside it, their programs, development scores and gaps. Saved per-image metrics
and trajectories are available on demand. Probe reruns have a separate budget of
nine; ordinary file reads do not consume that budget.

Final selection uses qualifying new policies when available, otherwise all
completed new policies. It independently selects highest CLIPScore, highest
ImageReward, lowest LPIPS, lowest NFE and lowest mean ranking objective. Repeated
categories share one frozen program pair; each unique pair is evaluated once on
the held-out set. Single is the lowest-development-objective representative.

Outputs include `rounds/`, `strategies/`, `evaluations/`, `diagnostics/`,
`leaderboard.json`, `state.json`, `selected_representatives.json` and
`test/results.json`. Each archived and frozen policy contains both source files.
Held-out scores never feed proposal generation or policy selection.

The root README documents the shared metrics and target values; the experiment
JSON controls them.
