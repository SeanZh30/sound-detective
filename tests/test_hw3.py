"""Checks that address leakage, baseline equivalence, and the gating mechanism."""
import numpy as np
import torch
from training.hw3.model import TemporalMLP
from training.hw3.prepare import extract
from training.features import extract_features

def test_global_features_match_hw2():
    sr=22050
    rng=np.random.default_rng(20)
    audio=rng.normal(0,.1,sr*3).astype('float32')
    np.testing.assert_allclose(extract(audio,sr)[:168],extract_features(audio,sr),rtol=1e-5,atol=1e-5)

def test_official_short_clips_not_discarded():
    x=extract(np.ones(1323,dtype='float32')*.1,22050)
    assert x.shape==(840,) and np.isfinite(x).all()

def test_fixed_and_initial_gate_are_identical():
    torch.manual_seed(1)
    fixed=TemporalMLP('fixed').eval()
    gated=TemporalMLP('gated').eval()
    gated.load_state_dict(fixed.state_dict(),strict=False)
    x=torch.randn(8,840)
    a,gate=gated(x,return_gate=True)
    torch.testing.assert_close(a,fixed(x))
    torch.testing.assert_close(gate,torch.full((8,1),.5))
    a.sum().backward()
    assert gated.gate.weight.grad.abs().sum()>0
    assert gated.local_layer.weight.grad.abs().sum()>0
    assert gated.global_layer.weight.grad.abs().sum()>0

def test_baseline_ignores_local_features():
    model=TemporalMLP('baseline').eval()
    x=torch.randn(8,840); altered=x.clone();altered[:,168:]=99
    torch.testing.assert_close(model(x),model(altered))

def test_variants_learn_from_gradients():
    torch.set_num_threads(2)
    x=torch.randn(16,840); y=torch.arange(16)%10
    for variant in ['baseline','concat','fixed','gated']:
        torch.manual_seed(42)
        model=TemporalMLP(variant,dropout=0)
        opt=torch.optim.SGD(model.parameters(),lr=.05)
        initial=torch.nn.functional.cross_entropy(model(x),y).item()
        for _ in range(25):
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();opt.step()
        assert torch.nn.functional.cross_entropy(model(x),y).item()<initial*.5

def test_purge_retains_test_but_isolates_all_sources():
    from training.hw3.train import split_masks
    folds=np.array([1,1,2,2,3,3,4])
    sources=np.array(['shared-test','a','shared-test','shared-val','shared-val','shared-test','clean'])
    m=split_masks(folds,sources,1)
    assert np.flatnonzero(m['test']).tolist()==[0,1]
    assert np.flatnonzero(m['validation']).tolist()==[3]
    assert np.flatnonzero(m['train']).tolist()==[6]
