from vireon_trp import TRPToyModel

def test_toy_model_runs():
    model = TRPToyModel(seed=0)
    R, P, D = model.run(steps=50, u=0.02)
    assert len(R) == 50
    assert len(P) == 50
    assert len(D) == 50
    assert P[-1] >= 0.0
