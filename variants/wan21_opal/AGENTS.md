# Independent Wan video search

This is the user-authorized video migration of the pinned OPAL image search.
Changes belong in this directory. Preserve image search, existing video/baseline implementations and their running experiments.

Keep the common, first-round and later-round prompt structures. Policies edit
only `extract_signals(tensors, state)` and `choose(observation, signals, state)`.
The framework owns observations, CFG, scheduler, residual boundaries and metrics.

Use the three-metric sqrt-NFE loss, initial loss=1, uncapped PSNR and no
normalization. Pareto has five objectives. Final selection requires all four:
mean PSNR>25.1022, mean SSIM>0.8520, mean LPIPS<0.0904 AND mean NFE<=44. Initial is
excluded. Per-round Pareto selection and feedback remain unchanged. Only after all
search rounds, select lowest-LPIPS, fastest, and lowest-existing-loss representatives
among all qualifying new strategies. Categories may share a strategy; test unique
strategies once, without selecting on test scores. If none qualify, select the same three categories from all completed new strategies and report no qualifying strategy.

Keep development-only tests outside the project. GPU generation and paid Agent
calls require an execution request. CPU checks run inside tmux in the existing
environment. Reference import and check do not generate videos or call the Agent.

Policies use the wan21-opal-step-signals-v3 interface. Run signal extraction and
choose once per sampling step with one state per video. Apply the shared action
to both CFG branches and keep two separate residual caches. A step uses 0 or 2 NFE.
Record the shared observation and decision plus both branch executions.

Development strategy evaluation first scores the fixed first 20 prompts in configuration
order. Stop only when their mean PSNR < 17 AND SSIM < 0.6 AND LPIPS > 0.3;
otherwise evaluate the remaining prompts. Both GPUs finish their ten screening
samples before continuation. Reference generation, probes and final testing do not
use screening. Keep partial scores in screening.json, label early_stopped feedback
in new_feedback, and provide direct material paths. These submissions consume round
slots without replacement, remain available for revision and diagnosis, and never
enter full-development Pareto or final selection. Existing completed results are retained.
