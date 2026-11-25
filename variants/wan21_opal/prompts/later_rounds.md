# Subsequent round: revise or explore using measured feedback

You receive the cumulative Pareto frontier updated after the preceding round,
with complete strategy code and mean evaluation scores, plus new feedback.
You also receive every qualifying strategy outside the frontier under
qualified_outside_pareto. Together, these lists provide the full frontier and
all qualifying strategies without duplicated code.

Choose how to develop the next proposals:

1. Modify a supplied Pareto strategy to improve its signals or decision rules.
2. Explore a new direction by designing different signals, formulas or rules
   informed by the available evidence.
3. Read another historical strategy on demand and develop it further, including
   strategies outside the current frontier.

Choose or combine these approaches freely. For each package, identify its
evidence and hypothesis. Name the source strategy as parent when revising one;
for a fresh direction, name a completed strategy as the comparison parent
without needing to inherit its implementation.

Submit requested_new_strategies complete packages (5 at the start of each later
round). Failed attempts consume slots. Each round allows at most three replacement
submission batches for failed slots, then advances using its successful strategies.

You may request a diagnostic before submission by writing diagnostic_request.json
with kind="probe" and probe_path="probe.py". The probe defines
extract_signals(tensors, state) to investigate a specific question.
Use the findings to continue the same round and submit the requested packages.

The current baseline development-mean quality scores are PSNR: 25.1022,
SSIM: 0.8520, LPIPS: 0.0904. Your goal is to reduce full backbone evaluations
and generation time while strictly improving all three quality scores.
Final qualification requires all four development-mean conditions:
    PSNR > 25.1022 AND SSIM > 0.8520 AND LPIPS < 0.0904 AND NFE <= 44. Prioritize meeting the quality requirements,
    then aim to meet the NFE target while maintaining those quality levels.
Use measured scores and gaps to guide further revisions.
Quality-ineligible strategies remain valid search evidence. Final representatives
are selected after all rounds; each round continues to use Pareto feedback.
