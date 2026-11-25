from types import SimpleNamespace


class WholeBackboneCache:
    def __init__(self, generator, extract, choose):
        self.generator, self.model = generator, generator.diffusion_pipeline.transformer
        self.extract, self.choose = extract, choose
        self.original_forward = self.model.forward
        self.model.forward = self.forward

    def reset(self, row, control=False):
        self.row, self.control = row, control
        self.state = {'scalars': {}, 'tensors': {}}
        self.residual = self.previous_latent = self.previous_feature = None
        self.step = self.nfe = self.age = 0
        self.reference_sigma = None
        self.events = []

    def forward(self, **kwargs):
        import torch
        from reactive_samplers.policy_observations import analyze
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
        if self.control:
            return self.original_forward(**kwargs)
        latent = kwargs['hidden_states']
        feature = self.model.x_embedder(latent)
        sigmas = self.generator.diffusion_pipeline.scheduler.sigmas
        sigma, target = float(sigmas[self.step]), float(sigmas[self.step + 1])
        observation = analyze(SimpleNamespace(latent=latent, input_feature=feature,
            reference_latent=self.previous_latent, reference_feature=self.previous_feature,
            cache_observation={'sigma': sigma, 'step_delta_sigma': sigma - target,
                'since_real_delta_sigma': 0.0 if self.reference_sigma is None else self.reference_sigma - sigma,
                'consecutive_skips': self.age, 'has_reference': self.residual is not None}))
        tensors = {'latent': latent, 'feature': feature,
                   'previous_latent': self.previous_latent, 'previous_feature': self.previous_feature}
        tensors = {k: None if v is None else v.detach().clone() for k, v in tensors.items()}
        signals = self.extract(tensors, self.state)
        action = self.choose(observation, signals, self.state['scalars'])
        reuse = action.skip and self.residual is not None
        if reuse:
            timestep = kwargs['timestep'].to(feature.dtype) * 1000
            guidance = kwargs['guidance'].to(feature.dtype) * 1000
            temb = self.model.time_text_embed(timestep, guidance, kwargs['pooled_projections'])
            output = self.model.proj_out(self.model.norm_out(feature + self.residual, temb))
            prediction = Transformer2DModelOutput(sample=output) if kwargs.get('return_dict', True) else (output,)
            self.age += 1
        else:
            captured = []
            hook = self.model.norm_out.register_forward_pre_hook(
                lambda module, inputs: captured.append((inputs[0] - feature).detach()))
            try:
                prediction = self.original_forward(**kwargs)
            finally:
                hook.remove()
            self.residual = captured[0]
            self.nfe += 1
            self.age = 0
            self.reference_sigma = sigma
        self.events.append({'step': self.step, 'sigma': sigma, 'reuse_cache': reuse,
                            'nfe': self.nfe, 'observation': observation, 'signals': signals})
        self.previous_latent, self.previous_feature = latent.detach().clone(), feature.detach().clone()
        self.step += 1
        return prediction
