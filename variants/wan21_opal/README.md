# Wan policy search

Wan2.1-T2V-1.3B uses the official CFG and UniPC path: 50 steps, guidance 6, shift
8, 832x480, 81 frames at 16 FPS, batch size one. The committed VBench split is
124 development and 820 held-out prompts, with prompt-specific seeds.

```bash
python -m variants.wan21_opal.search check experiments/wan21-vbench124-820/search.json
python -m variants.wan21_opal.search run experiments/wan21-vbench124-820/search.json
python -m variants.wan21_opal.search test experiments/wan21-vbench124-820/search.json
```

`check` is CPU-only; `run --resume` continues the same recorded search. The
root README describes the RunPod Agent relay and Slurm workers. Each GPU owns a
persistent model process and generates assigned prompts serially. The controller
alone updates the archive and Pareto frontier.

One policy decision controls both CFG branches at each node. Branches keep
separate residual caches. NEW costs two full evaluations; REUSE costs zero, so
the all-NEW control has NFE 100. The Agent edits only `signals.py` and
`updater.py`, with one independent policy state per video.

Nine rounds offer 10 + 8x5 candidate slots and a separate nine-request probe
budget. Development evaluation first scores the fixed first 20 prompts and
stops only if PSNR < 17, SSIM < 0.6 and LPIPS > 0.3 all hold. Early-screened
policies retain their partial feedback but do not enter full-development Pareto
or final selection. References, probes and held-out testing are not screened.

Qualification requires development PSNR > 25.1022, SSIM > 0.8520, LPIPS < 0.0904
and NFE <= 44. Final selection independently minimizes LPIPS, generation time
and the ranking objective in the qualifying pool, or all completed new policies
when none qualify. Repeated categories share one frozen pair and one held-out
evaluation. Single uses the lowest-development-objective representative.

`prepare.py` can import matched all-NEW references or construct an independent
configuration from explicit prompt files. For the paper split pass
`experiments/wan21-vbench124-820/exploration_prompts.json` and
`evaluation_prompts.json` through its `--exploration-prompts` and
`--evaluation-prompts` arguments. Existing run-local results remain outside Git.
