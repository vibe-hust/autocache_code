"""Official FLUX SeaCache gating on Qwen-Image-2.1 image tokens."""
from .util_seacache import apply_sea_with_scheduler, rel_l1
from baselines.qwen21_cache.qwen21_batch_cache import Qwen21BatchCache


class Qwen21SeaCache(Qwen21BatchCache):
    def __init__(self, transformer, scheduler, steps, threshold, grid):
        # Reuse hook installation and grouped Prefix KV calls, not TeaCache gating.
        super().__init__(transformer, 'teacache', steps, threshold)
        self.method = 'seacache'
        self.scheduler, self.grid = scheduler, grid

    def _block(self, index, original, *args, **kwargs):
        hidden = kwargs['hidden_states']
        if index == 0:
            step = len(self.batch_events)
            batch = hidden.shape[0]
            if step == 0:
                if kwargs.get('kv_cache_mode') != 'extract':
                    raise ValueError('SeaCache requires official Qwen Prefix KV prefill')
                self.accumulated = [0.] * batch
                self.counts, self.block_counts = [0] * batch, [0] * batch
                self.residuals = [None] * batch
                self.events = [[] for _ in range(batch)]
            self.target_mask = kwargs['target_token_mask']
            self.input = hidden[:, self.target_mask].detach()
            block = self.model.transformer_blocks[0]
            modulation = kwargs['modulation'].chunk(2, dim=-1)[0]
            signal, _ = block._modulate(block.img_norm1(hidden), modulation, self.target_mask)
            signal = signal[:, self.target_mask].detach()
            distances = [None] * batch
            forced = step in (0, self.steps - 1)
            if forced:
                self.accumulated = [0.] * batch
                self.reuse = [False] * batch
            else:
                signal = apply_sea_with_scheduler(
                    signal.reshape(batch, *self.grid, signal.shape[-1]),
                    self.scheduler, step, power_exp=2., dims=(-2, -3), norm_mode='mean',
                ).reshape_as(signal)
                distances = [rel_l1(signal[i], self.previous_signal[i]) for i in range(batch)]
                self.accumulated = [a + d for a, d in zip(self.accumulated, distances)]
                self.reuse = [a < self.threshold for a in self.accumulated]
            # Upstream stores the raw signal at the first/last node, filtered otherwise.
            self.previous_signal = signal
            self.refresh_indices = [i for i in range(batch) if not self.reuse[i]]
            for i in range(batch):
                self.counts[i] += int(not self.reuse[i])
                self.events[i].append({
                    'step': step, 'sigma': float(self.scheduler.sigmas[step]),
                    'signal_delta': distances[i], 'accumulated_distance': self.accumulated[i],
                    'reuse_cache': self.reuse[i], 'nfe': self.counts[i],
                })
            for i in self.refresh_indices:
                self.accumulated[i] = 0.
            self.batch_events.append(list(self.reuse))

        output = hidden.clone()
        if self.refresh_indices:
            output[self.refresh_indices] = self._selected(original, kwargs, self.refresh_indices)
            for i in self.refresh_indices:
                self.block_counts[i] += 1
        if index == 0:
            for i, reuse in enumerate(self.reuse):
                if reuse:
                    output[i, self.target_mask] = self.input[i] + self.residuals[i]
        if index == len(self.originals) - 1:
            for i in self.refresh_indices:
                self.residuals[i] = (output[i, self.target_mask] - self.input[i]).detach()
        return output
