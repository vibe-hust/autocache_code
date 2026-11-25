# Independent OPAL search

This directory is the explicitly authorized tensor-signal search variant. Preserve shared sampling/cache/scheduler mechanisms and the separate video implementation. Do not revive retired training or editable-analysis exploration.

Agent policy packages contain only signals.py and updater.py. Both functions are
editable under this variant's restricted interface; freeze and evaluate them together.
The controller owns data, seeds, model, precision, evaluation, loss and Pareto selection.
Use uncapped unnormalized loss; assign initial all-NEW loss=1; identical-image formula
loss=0. The numerator is sqrt(NFE). Initial slots are 10 then eight rounds of 5; failures consume slots, with at most
three replacement submission batches per round. Advance using successful strategies;
per-round Pareto selection and feedback remain unchanged.

No default tensor dumps. Existing nine scalars, added scalar signals and decisions are
logged at each node. Optional probes are separate measured experiments.
Only probe reruns consume diagnostic_budget (default 9 across the search).
Existing file reads and trajectory diagnostics do not consume this budget.
Successful diagnostic calls do not consume the per-round Agent attempt limit.
GPU generation and paid Agent calls require an execution request; check is CPU-only.
Keep development-only checks outside the project, as required by the parent instructions.

Final qualification requires all development means: NFE <= 12.7, CLIPScore > 30.7619,
ImageReward > 0.5325, PSNR > 29.9745, SSIM > 0.9186 and LPIPS < 0.0918.
Archive and Pareto feedback retain nonqualifying strategies with gaps.
After all search rounds, use qualifying new strategies if any; otherwise use
all completed new strategies and report zero qualifying strategies. Select
highest-CLIPScore, highest-ImageReward, lowest-LPIPS, lowest-NFE and lowest-loss
representatives independently. Categories may share a strategy; test each
unique strategy once. Exclude initial and failed strategies.

FLUX3 uses the same engine with its own configured fidelity/NFE targets, 50 nodes,
CFG=4 and original prompts. Qwen and FLUX targets are generated from the same
quality_targets configuration used by execution. Nonpositive ranking denominators
are null on both splits; average valid objective values while retaining every
sample for the other metrics. Qwen TeaCache keeps direct accumulation.
