"""Read-only fixed-episode projection and carrier diagnostics for local TAR fits."""

import hashlib

import torch


class EncoderMonitor:
    def __init__(self, model, episode):
        self.model = model
        self.episode = episode
        self.initial = model.continuous.detach().double().cpu().clone()
        self.initial_frequencies = model.frequencies.detach().double().cpu().clone()
        self.gradient_norm = None
        self.hook = None
        if model.continuous.requires_grad:
            self.hook = model.continuous.register_hook(self.capture_gradient)
        with torch.no_grad():
            h = model.compile_episode(episode)[0]
            n, m = episode.values.shape
            self.visible = episode.visible.to(h.device)
            self.initial_carriers = h[:n, :m][self.visible].double().cpu().clone()

    def capture_gradient(self, gradient):
        # Parameter hook runs before the trainer's global gradient clipping.
        self.gradient_norm = float(gradient.detach().double().norm())

    @torch.no_grad()
    def measure(self):
        weight = self.model.continuous.detach().double().cpu()
        singular = torch.linalg.svdvals(weight)
        h = self.model.compile_episode(self.episode)[0]
        n, m = self.episode.values.shape
        carriers = h[:n, :m][self.visible].double().cpu()
        return dict(
            projection_role=("shared_W_enc" if self.model.config.value_encoding ==
                             "unified_constant_weight" else "legacy_numeric_projection"),
            trainable=self.model.continuous.requires_grad,
            weight_sha256=hashlib.sha256(
                self.model.continuous.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
            drift_frobenius=float((weight - self.initial).norm()),
            relative_drift=float((weight - self.initial).norm() / self.initial.norm()),
            sigma_min=float(singular.min()), sigma_max=float(singular.max()),
            weight_norm=float(weight.norm()),
            frequency_drift=float((self.model.frequencies.detach().double().cpu()
                                   - self.initial_frequencies).norm()),
            visible_carrier_relative_drift=float((carriers - self.initial_carriers).norm()
                                                / self.initial_carriers.norm()),
        )

    def close(self):
        if self.hook is not None:
            self.hook.remove()
