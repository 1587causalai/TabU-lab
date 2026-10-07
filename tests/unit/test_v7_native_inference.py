"""Native inference rebuilds each supported checkpoint's recorded graph."""

import hashlib

import numpy as np
import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from tabu_lab.evaluation.tabarena.estimator import TabUV7Classifier, TabUV7Regressor
from tabu_lab.models.restoration_v7 import DualStreamConfig, V7Config, V7Model


@pytest.mark.parametrize("version,encoder", [("v7", "phi_lift"), ("v7.3", "phi_lift"), ("v7.3", "row_dual_stream")])
@pytest.mark.parametrize("classification", [False, True])
def test_native_estimator_loads_recorded_graph(tmp_path, version, encoder, classification):
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8)
            config = V7Config.for_version(
                    version,
                    value_encoder=encoder,
                    dual_stream=DualStreamConfig() if encoder == "row_dual_stream" else None,
                backbone=dict(width=128, layers=1, heads=4, ff_width=128, slots=4),
                rounds=2,
                unit_layers=0,
                center_chunk_size=16,
            )
            network = V7Model(config).double()
        path = tmp_path / "checkpoint.pt"
        torch.save(dict(model_version=version, config=config.as_dict(), model=network.state_dict()), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.DataFrame({"x": np.linspace(-1., 1., 12), "c": ["a", "b"] * 6})
        target = np.array([0, 1] * 6) if classification else np.linspace(-2., 2., 12)
        estimator_type = TabUV7Classifier if classification else TabUV7Regressor
        estimator = estimator_type(checkpoint_path=path, checkpoint_sha256=digest).fit(frame, target)
        assert estimator.network_.config == config
        prediction = estimator.predict(frame.iloc[:3])
        assert prediction.shape == (3,) and np.isfinite(prediction).all()
        if classification:
            probability = estimator.predict_proba(frame.iloc[:3])
            assert probability.shape == (3, 2) and np.isfinite(probability).all()
            np.testing.assert_allclose(probability.sum(axis=1), 1.)
        assert estimator.predict(frame.iloc[:0]).shape == (0,)
    finally:
        torch.set_num_threads(old_threads)
