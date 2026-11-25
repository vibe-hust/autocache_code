# 1. Objective and scoring

Design reusable whole-backbone residual-cache strategies for Wan2.1-T2V-1.3B.
The current baseline development-mean quality scores are PSNR: 25.1022,
SSIM: 0.8520, LPIPS: 0.0904. Your goal is NFE approximately equal to or lower than this baseline,
strictly higher development-mean PSNR and SSIM,
and strictly lower development-mean LPIPS. Final qualification requires
all three quality conditions and development-mean NFE <= 44.
Use the framework's measured scores and gaps to guide revisions.
Loss remains feedback and selects only the final balanced representative.

For each video:
    loss = sqrt(NFE) / (PSNR
                  + 100*SSIM + 100 - 100*LPIPS)

Strategy loss is the mean of per-video losses over the fixed development set.
There is no PSNR cap or loss normalization. Only the initial all-NEW control
has an assigned loss of 1. JSON psnr=null means positive infinity, giving
per-video formula loss=0. A nonpositive denominator gives null loss; average valid per-video losses. All videos still contribute to other metrics.

Higher PSNR (pixel fidelity) and SSIM (structural fidelity) are better. Lower LPIPS
(perceptual distance), NFE (full backbone calls), generation time, and loss
are better. PSNR/SSIM/LPIPS compare with fixed matching all-NEW reference
videos. Metrics are computed on paired frames, then averaged per video.

Generation time is measured separately; it is not a term in the loss formula.

# 2. Information available

Read round_input.json in the current workspace.

The first round provides the initial all-NEW strategy and its measured
results. Subsequent rounds provide every strategy on the current Pareto
frontier, including its ID, complete code, mean loss, three mean quality
scores, mean NFE, and mean generation time. Every qualifying new strategy outside
the frontier is also supplied under qualified_outside_pareto. These two lists
form the default union without repeating strategy code or scores.

The Pareto frontier contains strategies that no other evaluated strategy
dominates: dominance requires being no worse on all three quality metrics,
NFE, and generation time, and strictly better on at least one.
This preserves multiple useful starting points rather than only the
lowest-loss strategy. Both qualifying and nonqualifying frontier strategies
are supplied with quality status and distances to the three quality boundaries and the NFE boundary.

Each request also provides the current round, requested number of new
strategies, remaining budget, and new feedback: paired comparisons,
failures, and newly completed diagnostics.

Available on demand at the supplied paths:
- Historical strategy code, submission descriptions, and cumulative scores.
- Per-video quality scores, NFE, time, and loss.
- Per-node observations, added signals, actions, conditions, and operands.
round_input.json lists direct per-strategy and per-sample paths under
on_demand.strategy_materials. Read trajectory_and_observations directly;
each JSONL line contains the shared observation and decision plus both branch executions.
history_strategies contains full code and mean scores for completed strategies
outside the default pool, using the same entry format as pareto.
new_feedback carries execution/submission errors and diagnostic answers.
Successful strategies are reviewed through their measured scores and material paths.

Read-only project code is available for inspecting the sampling and cache
implementation.

# 3. Mechanism and editable interface

NEW evaluates the backbone and caches:
    residual = output_feature - input_feature

REUSE adds the cached residual to the current input feature and executes
the output head. Decisions occur before the current full backbone prediction.

Each sampling step has conditional and unconditional branches. They use different
text conditioning but the same input latent and the same feature after patch embedding.
Signal extraction and choose run once per sampling step. Their single decision
applies to both branches. NEW runs both full backbone predictions and updates
their separate residual caches (2 NFE); REUSE uses both cached residuals (0 NFE).
CFG combines the two predictions, then the scheduler updates the latent once.

Each strategy contains two editable function bodies:
- signals.py: extract_signals(tensors, state)
  Returns a dictionary of finite Python scalars, booleans, or null.
- updater.py: choose(observation, signals, state)
  Returns SkipAction(False) for NEW or SkipAction(True) for REUSE.

The first request's interface object defines the nine base observations,
their formulas, tensor positions, and supported operations. Use the base
observations, derive new signals from exposed tensors, or combine both.

Tensor inputs are latent, feature, previous_latent, and previous_feature.
They are isolated per-video copies. Previous tensors refer to the adjacent
pre-update sampling node and are null initially.

extract_signals receives state containing 'scalars' and 'tensors'.
choose receives its 'scalars' dictionary only. State resets per video;
retain additional tensor history in state['tensors'] within the configured
per-video byte budget. Signal computation and history storage count toward runtime
and memory costs.

Edit only the two function bodies using the exposed interface. Imports,
extra helpers, module globals, model calls, file/network access, and
sample/prompt/seed lookup are outside this interface. Model weights,
cache payload and boundary, CFG, scheduler, and evaluation settings
remain framework-owned. Reference trajectories, future nodes, and quality
scores are not runtime policy inputs.

# 4. Analyze, diagnose, and propose

Connect observed behavior to a testable mechanism hypothesis. Inspect regions
with high NEW or REUSE frequency, including alternating patterns, and compare
behavior across samples. Use counterexamples to revise assumptions.

You may change signal formulas, thresholds, stage-dependent conditions, and
state usage. Build on any completed strategy or design a new rule, naming a
completed strategy as its comparison parent. For each proposal, briefly state
the change, supporting evidence, and hypothesis being tested.

Use diagnostics when existing results leave a specific question unresolved.
For example, does a small average change hide large changes at a few
positions, or could a new signal distinguish different refresh conditions?
Choose existing trajectory inspection or a custom tensor probe, then use
the result to guide the proposals.

Diagnostic options:
- trajectory: read selected recorded nodes without model calls.
- probe: run a custom probe.py implementing extract_signals(tensors, state)
  with separate diagnostic state per video and scalar outputs.

A probe replays only the requested samples with batch size 1 and collects
diagnostics at the selected nodes. Available GPUs share these samples. The framework
checks videos byte-for-byte and actions against the selected strategy's
recorded run. Probe costs are tracked separately from formal evaluations.
Large tensors are saved only when explicitly requested through capture_tensors.

# 5. Handoff, evaluation, and final selection

At each handoff, write exactly one of these files in the current workspace:

A. submission.json — submit the requested batch of strategies:

{"strategies":[{"id":"r01_s01","parent":"initial","path":"strategies/r01_s01",
"change":"short concrete change","evidence":"strategy IDs or observations",
"hypothesis":"what this change tests"}]}

Each strategy directory contains only signals.py and updater.py.
IDs must be unique across the experiment. Submit exactly
requested_new_strategies distinct packages.

B. diagnostic_request.json — request a diagnostic before submitting:

{"kind":"trajectory","strategy_id":"initial",
"sample_ids":["sample_id"],"steps":[3,4]}

For a custom probe, set kind="probe" and add "probe_path":"probe.py".
Optionally add "capture_tensors":["latent"] to save those tensors at the
selected nodes. Probe reruns consume the search-wide diagnostic budget
(default: 9). After receiving results, continue the same round and submit strategies when ready.

The framework checks, freezes, generates, and evaluates submitted packages.
You do not need to build your own benchmark to trial strategies before submission;
the framework handles generation and evaluation after submission.
Failed attempts consume submission slots. Each round starts with its configured
batch and allows at most three additional submission batches for failed slots.
After this limit, the framework advances using the successful archive; a round
with no new successes retains the previous frontier. No separate report, research notebook,
or self-reported check results are required.

Default schedule: 10 initial strategy slots in round 1, then 5 in each
of rounds 2–9. Every strategy is evaluated on the same development samples,
seeds, and configuration. Results are returned before the next round, using
the same Agent session.

All results are retained regardless of whether loss improves. Every round continues
to use the cumulative Pareto frontier and qualifying strategies outside it for feedback.
Only after all 9 rounds (50 initial slots plus bounded replacements), the framework
selects final representatives from all qualifying new strategies using development means:
best reference fidelity (lowest LPIPS), fastest qualifying policy (lowest generation
time), and balanced policy (lowest existing loss). Each category is selected independently;
ties use lower NFE, then lower generation time, then strategy ID. Categories may select
the same strategy. Unique strategies are evaluated once on the independent test set;
test results never reselect representatives. If none qualify, select these same
three categories from all completed new strategies and explicitly report that
no strategy qualified. Retain the Pareto frontier, measured means, and quality gaps;
qualification requirements and search budgets stay unchanged. The initial all-NEW control
is excluded from final selection.
