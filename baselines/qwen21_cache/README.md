# Qwen-Image-2.1 paper baselines

Run from the repository root:

```bash
python -m baselines.qwen21_cache.run_panel experiments/qwen21-drawbench40-baselines/generation.json
```

The panel generates paired official, TeaCache, DiCache and SeaCache outputs with
40 nodes, guidance one and the committed DrawBench prompts/seeds. Thresholds are
0.08/0.40/0.30. Qwen TeaCache retains direct relative-change accumulation.
DiCache warmup ratio is 0.20 (nine nodes), probe depth one block.
`Qwen21BatchCache` remains shared with SeaCache and GenEval 2 even though paper
evaluation uses one sample per GPU.
