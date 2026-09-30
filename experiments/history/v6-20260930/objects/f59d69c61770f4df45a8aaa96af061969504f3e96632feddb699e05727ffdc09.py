import random
BRANCH_CONFIG = dict(probability_v6=0.5, probability_v55=0.5, seed=2026092901, mode='independent_Bernoulli_per_update', selected_loss_weight=1.0, forwards=1, optimizer_steps=1)
class BranchSampler:
    def __init__(self, saved=None):
        self.rng=random.Random(BRANCH_CONFIG['seed'])
        if saved is not None:self.rng.setstate(saved)
    def draw(self):return 'v6' if self.rng.random()<0.5 else 'v55'
    def state(self):return self.rng.getstate()
