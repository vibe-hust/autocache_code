# First round: develop strategies from initial

Start from the initial all-NEW strategy and its measured results. Design an
initial set of distinct NEW/REUSE rules in updater.py, keeping signals.py unchanged and using the
available observations, tensors, implementation code and optional diagnostics.
You may propose new strategies from different perspectives and explore different directions.
Explain the evidence and mechanism hypothesis behind each proposal.
{{TARGETS}}
Use the framework's returned scores to guide later revisions.
Final category selection happens only after all search rounds; per-round Pareto feedback is unchanged.

Submit requested_new_strategies complete packages (10 at the start of round 1), using initial as the
comparison parent. If this round includes diagnostic or replacement calls,
use the newly returned feedback to complete the outstanding proposals.

Final representatives use qualifying new strategies if any; otherwise use all
completed new strategies. Select highest CLIPScore, highest ImageReward,
lowest LPIPS, lowest NFE and lowest loss independently. Categories may
share a strategy; test each unique strategy once.
