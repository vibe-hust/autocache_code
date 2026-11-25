"""Per-image TeaCache/DiCache decisions with grouped official Qwen21 blocks."""
import torch
from .qwen21_cache import Qwen21Cache


def relative_rows(current, previous):
    dims=tuple(range(1,current.ndim))
    return (current.float()-previous.float()).abs().mean(dims)/previous.float().abs().mean(dims).clamp_min(1e-8)


class Qwen21BatchCache(Qwen21Cache):
    """One batch per context; weights are shared, decisions and histories are not."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.batch_events=[]
        self.batch_block_calls=0

    def _selected(self, original, kwargs, indices):
        batch=kwargs['hidden_states'].shape[0]
        if len(indices)==batch:
            self.batch_block_calls+=1
            return original(**kwargs)
        sub=dict(kwargs)
        sub['hidden_states']=kwargs['hidden_states'][indices]
        # The last modulation row is the shared t=0 prefix row, not an image.
        modulation=kwargs['modulation']
        sub['modulation']=torch.cat([modulation[indices],modulation[-1:]],dim=0)
        for key in ('attention_mask','key_valid'):
            value=kwargs.get(key)
            if value is not None:
                if not isinstance(value,torch.Tensor):
                    raise ValueError('Grouped cache supports the official tensor attention processor only')
                sub[key]=value[indices]
        cache=kwargs.get('layer_cache')
        if cache is not None:
            if kwargs['kv_cache_mode']!='cached':
                raise ValueError('Prefix prefill must include the complete batch')
            from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVLayerCache
            selected=QwenImage21KVLayerCache()
            k,v=cache.get();selected.store(k[indices],v[indices])
            sub['layer_cache']=selected
        self.batch_block_calls+=1
        return original(**sub)

    def _block(self,index,original,*args,**kwargs):
        if args:
            raise ValueError('Official Qwen21 blocks must receive keyword arguments')
        hidden=kwargs['hidden_states']
        if index==0:
            step=len(self.batch_events)
            if step>=self.steps:
                raise RuntimeError('Use a fresh cache context for each batch')
            batch=hidden.shape[0]
            if step==0:
                if kwargs.get('kv_cache_mode')!='extract':
                    raise ValueError('Official prefix KV prefill is required')
                self.accumulated=torch.zeros(batch,device=hidden.device)
                self.counts=[0]*batch
                self.block_counts=[0]*batch
                self.histories=[[] for _ in range(batch)]
                self.probe_histories=[[] for _ in range(batch)]
                self.events=[[] for _ in range(batch)]
            self.target_mask=kwargs['target_token_mask']
            self.input=hidden[:,self.target_mask].detach()
            forced=step in (0,self.steps-1)
            signal_delta=None
            if self.method=='teacache':
                block=self.model.transformer_blocks[0]
                mod1=kwargs['modulation'].chunk(2,dim=-1)[0]
                signal,_=block._modulate(block.img_norm1(hidden),mod1,self.target_mask)
                signal=signal[:,self.target_mask].detach()
                if not forced:signal_delta=relative_rows(signal,self.previous_signal)
                self.previous_signal=signal
                output=None
            else:
                output=self._selected(original,kwargs,list(range(batch)))
                self.block_counts=[v+1 for v in self.block_counts]
                probe=output[:,self.target_mask].detach()
                forced=forced or step<=int(self.warmup_ratio*self.steps)
                if not forced:signal_delta=relative_rows(probe,self.previous_probe)
                self.previous_probe=probe
                self.probe_delta=probe-self.input
            if signal_delta is not None:self.accumulated+=signal_delta
            self.reuse=(self.accumulated < torch.as_tensor(self.threshold,device=hidden.device))
            if forced:self.reuse.zero_()
            self.refresh_indices=[i for i in range(batch) if not bool(self.reuse[i])]
            self.accumulated[self.refresh_indices]=0
            if self.method=='teacache':
                output=hidden.clone()
                if self.refresh_indices:
                    output[self.refresh_indices]=self._selected(original,kwargs,self.refresh_indices)
                    for i in self.refresh_indices:self.block_counts[i]+=1
            for i in range(batch):
                reuse=bool(self.reuse[i])
                if reuse:
                    delta=self.histories[i][-1]
                    if self.method=='dicache' and len(self.histories[i])==2:
                        before,latest=self.probe_histories[i]
                        denominator=(latest-before).float().abs().mean().clamp_min(1e-8)
                        gamma=((self.probe_delta[i]-before).float().abs().mean()/denominator).clamp(1,1.5)
                        delta=self.histories[i][0]+gamma.to(delta.dtype)*(self.histories[i][1]-self.histories[i][0])
                    output[i,self.target_mask]=self.input[i]+delta
                else:self.counts[i]+=1
                self.events[i].append(dict(event='step',step=step,nfe=self.counts[i],reuse_cache=reuse,
                    backbone_evaluated=not reuse,probe_blocks=int(self.method=='dicache'),
                    signal_delta=None if signal_delta is None else float(signal_delta[i]),
                    accumulated_distance=float(self.accumulated[i])))
            self.batch_events.append([bool(x) for x in self.reuse])
        else:
            output=hidden.clone()
            if self.refresh_indices:
                output[self.refresh_indices]=self._selected(original,kwargs,self.refresh_indices)
                for i in self.refresh_indices:self.block_counts[i]+=1
        if index==len(self.originals)-1:
            for i in self.refresh_indices:
                self.histories[i].append((output[i,self.target_mask]-self.input[i]).detach())
                self.histories[i]=self.histories[i][-2:]
                if self.method=='dicache':
                    self.probe_histories[i].append(self.probe_delta[i])
                    self.probe_histories[i]=self.probe_histories[i][-2:]
        return output
