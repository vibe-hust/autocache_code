# Baselines reported in the paper

Qwen and FLUX3: TeaCache, DiCache, SeaCache and MagCache.
Wan1.3B: locally rerun TeaCache, DiCache and SeaCache. Wan MagCache-slow,
NaviCache-mid and ERTACache rows marked with a dagger are cited results.
FLUX.1-dev also includes DPCache and ECAD Fast.

Retained upstream Wan packages provide the official generation path; FLUX/Wan
TeaCache and DiCache source provide the algorithms used by the adapters.
Qwen adapters are in `qwen21_cache/` and `variants/seacache/`. MagCache uses
the measured Diffusers-hook adapter in `variants/magcache/`. Copyright/license
files remain with retained sources. Model weights and environments are external.
BudCache and the recorded 20/28-step comparison entries are included.
