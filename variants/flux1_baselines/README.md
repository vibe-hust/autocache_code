# Supplementary FLUX.1-dev baselines

These are the measured source-run adapters. Their source config fixes 50 steps,
256×256 and guidance 5. DPCache uses K=25 and its own calibration matrix.
ECAD Fast uses its released 20-step schedule. They do not change the main
Qwen/FLUX3/Wan settings.

Install the original ECAD package used by these adapters in its separate
compatible environment: `pip install git+https://github.com/AniAggarwal/ecad@876ff68`.
DPCache's small source package and Apache-2.0 license are in `baselines/DPCache`.
The runner accepts an existing paired FLUX.1 source run and calibrated matrix.
The released ECAD Fast schedule is committed in `experiments/flux1-drawbench20/`.
Model weights remain external resources.
