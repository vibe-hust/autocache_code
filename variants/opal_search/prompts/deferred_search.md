# Deferred-evaluation ablation

Submit all nine proposal rounds before any initial or candidate image generation or measurement.
You receive initial all-NEW code and previous accepted proposal code, IDs, parents, change,
evidence and hypothesis. These packages are code-valid, not measured or quality-qualified.
Keep the same Agent session. Revisit earlier code and proposal reasoning to avoid duplicates.
Use mechanism reasoning and the interface, not claimed experimental evidence. Never invent scores.

No measured CLIPScore, ImageReward, PSNR, SSIM, LPIPS, NFE, latency, loss, actions,
observations, Pareto membership or quality gaps are supplied during proposal rounds.
The fixed target scores and quality thresholds below are unchanged reference objectives,
not measurements of this run. Initial all-NEW measured feedback is also delayed.
Do not read another experiment's results or run generation/scoring yourself.
Trajectory diagnostics and tensor probes cannot run during proposal collection.
You may queue diagnostic requests now; they execute automatically only after all
proposals are fixed and every accepted candidate has been evaluated. No diagnostic
results are returned during proposals, and no further proposals or feedback-driven
revisions occur after measurement.

# 1. Objective and scoring

Design reusable whole-backbone residual-cache strategies for the configured image backbone. {{TARGETS}}

For each image:
    loss = sqrt(NFE) / (CLIPScore + 100*ImageReward + PSNR
                  + 100*SSIM + 100 - 100*LPIPS)

Strategy loss is the mean of per-image losses over the fixed development set.
There is no PSNR cap or loss normalization. Only the initial all-NEW control
has an assigned loss of 1. JSON psnr=null means positive infinity, giving
per-image formula loss=0. A nonpositive denominator gives null loss; average only valid losses. All images still contribute to other metrics.

Higher CLIPScore (prompt alignment), ImageReward (human preference), PSNR
(pixel fidelity), and SSIM (structural fidelity) are better. Lower LPIPS
(perceptual distance), NFE (full backbone calls), generation time, and loss
are better. PSNR/SSIM/LPIPS compare with fixed matching all-NEW reference
images. ImageReward can be negative.

Generation time is measured separately; it is not a term in the loss formula.

# 3. Mechanism and editable interface

NEW evaluates the backbone and caches:
    residual = output_feature - input_feature

REUSE adds the cached residual to the current input feature and executes
the output head. Both actions execute the original scheduler update.
Signal extraction and choose run once per image at each sampling node, before
the current full backbone prediction. Each image has independent state and cache.

Each strategy contains two editable function bodies:
- signals.py: extract_signals(tensors, state)
  Returns a dictionary of finite Python scalars, booleans, or null.
- updater.py: choose(observation, signals, state)
  Returns SkipAction(False) for NEW or SkipAction(True) for REUSE.

The first request's interface object defines the nine base observations,
their formulas, tensor positions, and supported operations. Use the base
observations, derive new signals from exposed tensors, or combine both.

Tensor inputs are latent, feature, previous_latent, and previous_feature.
They are isolated per-image copies. Previous tensors refer to the adjacent
pre-update node and are null initially.

extract_signals receives state containing 'scalars' and 'tensors'.
choose receives the same 'scalars' dictionary only. State resets per image;
retain additional tensor history in state['tensors'] within the configured
byte budget. Signal computation and history storage count toward runtime
and memory costs.

Edit only the two function bodies using the exposed interface. Imports,
extra helpers, module globals, model calls, file/network access, and
sample/prompt/seed lookup are outside this interface. Model weights,
cache payload and boundary, Prefix KV, scheduler, and evaluation settings
remain framework-owned. Reference trajectories, future nodes, and quality
scores are not runtime policy inputs.

# Proposal handoff and fixed budgets

Read round_input.json. Submit exactly requested_new_strategies packages in submission.json:
{"strategies":[{"id":"r01_s01","parent":"initial","path":"strategies/r01_s01",
"change":"concrete change","evidence":"code IDs or mechanism reasoning; no invented measurements",
"hypothesis":"what the rule is intended to test"}]}

Each package contains signals.py and updater.py only. IDs are unique across this run.
Parent names initial or an already accepted proposal; it need not have measured results.
Only syntax, restricted interface, package and existing duplicate-code checks run now.
Code/interface errors are returned for repair. Failed submissions consume slots/calls
under the existing rules: 10 initial slots in round 1, then 5 in each of rounds 2-9;
at most three additional submission batches for failed slots per round and the same
per-round Agent call limit. Advance with the accepted archive after bounded replacements.
Instead of submission.json, you may write diagnostic_request.json to queue a request:
{"kind":"trajectory","strategy_id":"initial","sample_ids":["sample_id"],"steps":[0,1]}
Use diagnostic_sample_ids from round_input.json and valid zero-based sampling steps.
The strategy must be initial or an accepted proposal. For kind="probe", also provide
"probe_path":"probe.py" implementing extract_signals(tensors, state); optionally add
"capture_tensors":["latent"]. The framework preserves this code when queuing it.
Only probe requests consume remaining_diagnostic_budget (nine across the search).
A queued request does not fill candidate slots or consume a proposal-call attempt.
You receive only a queue acknowledgement; continue the same round and submit the
requested strategies. Invalid requests return interface errors under the existing
call limit. Do not run a diagnostic, generation or scoring yourself.

After round 9, the framework generates the initial all-NEW reference and evaluates
every accepted candidate on the same development prompts, seeds and generation settings.
Runtime failures remain failed evaluations; no new proposals or replacement rounds are
opened after measurement. This is deliberate: candidate proposals must not use measured feedback.
After candidate evaluation, execute queued diagnostics and archive their results.
A request targeting a failed candidate is archived as failed without adding proposals.
Then compute the same loss, Pareto archive and all-six-condition qualification gate.
Select independently highest CLIPScore, highest ImageReward, lowest LPIPS, lowest NFE
and lowest existing loss from qualifying new strategies if any; otherwise use all
completed new strategies and report zero qualifiers. Categories may share a strategy.
Exclude initial and failed candidates. Freeze the selected packages and test each unique
strategy once on the same independent test set; test scores never reselect representatives.
