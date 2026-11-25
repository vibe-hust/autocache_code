# Paper metrics

Image quality: CLIPScore, ImageReward, PSNR, SSIM and AlexNet LPIPS v0.1.
CLIPScore is `max(100*cos(image_embedding,text_embedding),0)` with OpenAI CLIP
ViT-B/32 and original prompts. Fidelity uses matching full-computation references.
Video fidelity uses all 81 decoded paired frames, averaged within video and then
equally over videos. These implementations are in `reactive_samplers/evaluation.py`
and `reactive_samplers/video/metrics.py`.

OpenAI CLIP source is retained as `clip/source`. Checkpoints remain external:
`clip/checkpoints/ViT-B-32.pt`, `image_reward/checkpoints/ImageReward.pt` and
`image_reward/checkpoints/med_config.json`. ImageReward can use a separate existing
Python environment through `metrics_python`; quality scoring is outside generation
timing. The operator TFLOPs counter is `reactive_samplers/operator_counter.py`.
