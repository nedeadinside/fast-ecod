import math

import numpy as np
import pytest
from fast_ecod import ECODScoreMethod, InductiveECOD

# Input layout is (n_features, n_samples)
N_FEATURES, N_SAMPLES = 3, 50


def make_data() -> np.ndarray:
    """
    Build gaussian matrix.

    :return: Matrix of shape (features, samples).
    """
    x = np.random.default_rng(0).normal(size=(N_FEATURES, N_SAMPLES))
    return x


def test_fit_returns_self():
    """
    Check that fitting returns the same model for chaining.
    """
    model = InductiveECOD()
    assert model.fit(make_data().tolist()) is model


def test_scores_shape():
    """
    Check that scoring yields one finite float per predicted sample.
    """
    model = InductiveECOD().fit(make_data().tolist())
    scores = model.decision_function(make_data()[:, :7].tolist())
    assert len(scores) == 7
    assert all(isinstance(s, float) and math.isfinite(s) for s in scores)


def test_default_method_is_max():
    """
    Check that scoring without a method matches the max method.
    """
    x = make_data().tolist()
    model = InductiveECOD().fit(x)
    assert model.decision_function(x) == model.decision_function(x, ECODScoreMethod.MAX)


def test_accepts_numpy():
    """
    Check that numpy arrays give the same scores as nested lists.
    """
    x = make_data()
    from_list = InductiveECOD().fit(x.tolist()).decision_function(x.tolist())
    from_numpy = InductiveECOD().fit(x).decision_function(x)
    assert from_list == from_numpy


def test_predict_before_fit_raises():
    """
    Check that scoring an unfitted model raises.
    """
    with pytest.raises(RuntimeError, match="not fitted"):
        InductiveECOD().decision_function([[1.0, 2.0]])


@pytest.mark.parametrize(
    "x",
    [
        [],
        [[1.0, 2.0], [3.0]],
        [[1.0, math.nan]],
        [[1.0, math.inf]],
    ],
    ids=["empty", "ragged", "nan", "inf"],
)
def test_fit_rejects_bad_input(x: list[list[float]]):
    """
    Check that fitting on malformed input raises.

    :param x: Malformed feature matrix.
    """
    with pytest.raises(ValueError):
        InductiveECOD().fit(x)


def test_feature_mismatch_raises():
    """
    Check that scoring with a different feature count raises.
    """
    model = InductiveECOD().fit(make_data().tolist())
    with pytest.raises(ValueError, match="features"):
        model.decision_function(make_data()[:2].tolist())
