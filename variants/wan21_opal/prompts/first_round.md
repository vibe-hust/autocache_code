# First round: develop strategies from initial

Start from the initial all-NEW strategy and its measured results. Design an
initial set of distinct signal-extraction and NEW/REUSE rules, using the
available observations, tensors, implementation code and optional diagnostics.
You may propose new strategies from different perspectives and explore different directions.
Explain the evidence and mechanism hypothesis behind each proposal.
The current baseline development-mean quality scores are PSNR: 25.1022,
SSIM: 0.8520, LPIPS: 0.0904. Your goal is to reduce full backbone evaluations
and generation time while strictly improving all three quality scores.
Final qualification requires all four development-mean conditions:
    PSNR > 25.1022 AND SSIM > 0.8520 AND LPIPS < 0.0904 AND NFE <= 44.
Use the framework's returned scores and gaps to guide later revisions toward
these goals while seeking lower NFE and generation time.
Final category selection happens only after all search rounds; per-round Pareto feedback is unchanged.

Submit requested_new_strategies complete packages (10 at the start of round 1), using initial as the
comparison parent. If this round includes diagnostic or replacement calls,
use the newly returned feedback to complete the outstanding proposals.
