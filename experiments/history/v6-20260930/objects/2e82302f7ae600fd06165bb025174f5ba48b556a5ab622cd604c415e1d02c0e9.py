"""Two forward maps, exactly one parameter set and one optimizer."""
from tabu_lab.models.restoration_v6 import V6Model
from tabu_lab.models.restoration_v55 import V55Model
from tabu_lab.models.restoration_v53.training import prepare_training_episode, score_prepared_episode, V53LossConfig

class DualModel(V6Model):
    def __init__(self, config):
        super().__init__(config, supervision='target_only')
        self.branch = 'v6'
    def forward_prepared(self, prepared, *, decode=True):
        if self.branch == 'v6':
            return V6Model.forward_prepared(self, prepared, decode=decode)
        if self.branch == 'v55':
            return V55Model.forward_prepared(self, prepared, decode=decode)
        raise ValueError(self.branch)

def prepare(model, inputs, request, truth, config):
    prepared=prepare_training_episode(model,inputs,request,truth,config)
    assert tuple(config.state_weights)==(0.,1.,0.,0.)
    assert len(prepared.visible.request.targets)==int(inputs.query.sum())
    assert len(inputs.query.any(0).nonzero())==1
    return prepared

def score(model,prepared,config,branch,decode=False):
    model.branch=branch
    return score_prepared_episode(model,prepared,config,decode=decode)
